import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial
import pickle
import time
import os
import argparse
import glob
import re
import sys
from find_smooth_submanifold import filter_and_refine, normalize_coeffs, get_basis_labels, combine_to_complex_equations
from slag_condition import compute_combined_fitness
from helper import assert_metric_psi_compatible, canonicalize_coeffs, format_array_with_commas, calculate_distance_matrix, dwork_points_path, load_points
from viz.fitness_pipeline import run_fitness_pipeline

jax.config.update('jax_default_matmul_precision', 'high')


def device_put_sharded(shards, devices):
    """Drop-in replacement for the deprecated jax.device_put_sharded."""
    mesh = Mesh(np.array(devices), ('x',))
    sharding = NamedSharding(mesh, P('x'))
    return jax.tree.map(
        lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards
    )

# -----------------------------------------------------------------------------
# 1. HYPERPARAMETERS
# -----------------------------------------------------------------------------
# Moduli of the quintic (complex; integer-real values map to legacy filenames psi0, psi10, ...).
PSI = 0+0j
SEED = 1024

# Edit this line if you're not using the Dwork-family naming convention,
# e.g. POINTS_FILE = "data/my_cicy.pkl"
POINTS_FILE = dwork_points_path(PSI, SEED)

# Metric used when compute the kahler form
# Options are 1. FS - Fubini-Study metric
#             2. k4_fermat - Ricci-flat metric with k = 4 from Headrick-Nassar energy-functional minimization. Fermat only.
#METRIC = 'FS'
METRIC = 'k4_fermat'

assert_metric_psi_compatible(METRIC, PSI)

# GA Parameters
POPULATION_SIZE = 800
GENOTYPE_SHAPE = (3, 250)
NUM_GENES = GENOTYPE_SHAPE[0] * GENOTYPE_SHAPE[1]
NUM_GENERATIONS = 40

#TRANSITION_GENERATION = 1600
TRANSITION_GENERATION = 999999

# --- Dynamic Speciation Parameters ---
TARGET_SPECIES_COUNT_MIN = 10
TARGET_SPECIES_COUNT_MAX = 25

SPECIATION_THRESHOLD_INIT = 1.77
SPECIATION_THRESHOLD_STEP = 0.02 # each gen *= (1 +/- step)
WARMUP_GENERATIONS = 300
COOLDOWN_GENERATIONS = 20
SPECIATION_MERGE_RATIO = 0.5  # Merge if distance smaller than threshold * ratio

TERRITORY_BUFFER_EXPLORE = 0.0
TERRITORY_BUFFER_EXPLOIT = 0.0

# Exploration Phase Settings
TOURNEY_SIZE_EXPLORE = 3
MUTATION_RATE_EXPLORE = 1.0 / NUM_GENES  # Higher rate
ETA_MUTATION_EXPLORE = 10.0
ETA_CROSSOVER_EXPLORE = 5.0
#SPECIATION_THRESHOLD_EXPLORE = 2.5
#SPECIES_SHARING_RADIUS_EXPLORE = 2.7

# Exploitation Phase Settings
TOURNEY_SIZE_EXPLOIT = 7
MUTATION_RATE_EXPLOIT = 0.5 / NUM_GENES  # Lower rate
ETA_MUTATION_EXPLOIT = 100.0
ETA_CROSSOVER_EXPLOIT = 30.0
#SPECIATION_THRESHOLD_EXPLOIT = 1.5 
#SPECIES_SHARING_RADIUS_EXPLOIT = 1.7

# Crossover and Mutation Parameters
# In this version the crossover has been turned off so this 
# parameter is not being used
CROSSOVER_RATE = 0.9

STAGNATION_THRESHOLD = 20  # Generations a species can go without improvement before being removed.
STAGNATION_SURVIVAL_RATIO = 0.9  # Always keep the species as long as their fitness is above the max fitness times this rate
SPECIES_ELITISM = 1        # Number of best individuals per species to carry over directly.

# --- Adaptive Step Size (1/5th success rule per species) ---
SIGMA_INIT = 1.0
SIGMA_MIN = 0.001
SIGMA_MAX = 5.0
SIGMA_INCREASE = 1.3   # on improvement
SIGMA_DECAY = 0.93     # on no improvement (targets ~1/5 success: 1.3 * 0.93^4 ≈ 0.97)
SIGMA_COOLDOWN = 3     # generations between sigma updates (lets effect propagate)

# Batching for Fitness Evaluation
FITNESS_MINI_BATCH_SIZE = 200
LOG_INTERVAL = 10

# Checkpointing
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_INTERVAL = 100

#MINSET_SIZE = 100000
#NEWTON_STEPS = 100
MINSET_SIZE = 10000
NEWTON_STEPS = 40

# Memory / Performance tuning hyperparameters.
# Controls how many points are processed in parallel during the distance
# calculation phase. Max value is the total number of points in the point
# cloud (e.g. 1,000,000). Lower if you hit XLA OOM.
DIST_CHUNK_SIZE = 50000



# JAX PRNG Key
key = jax.random.PRNGKey(1234)

# -----------------------------------------------------------------------------
# 2. CORE EVALUATION FUNCTIONS
# -----------------------------------------------------------------------------
@partial(jit, static_argnames=('k', 'n_refine_steps', 'metric'))
def calculate_fitness_for_one_individual(
    coeffs: jnp.ndarray,
    points_real: jnp.ndarray,
    psi: jnp.ndarray,
    k: int,
    n_refine_steps: int,
    metric: str = 'FS'
) -> jnp.float32:
    """
    Calculate fitness for one individual with automatic patch handling.

    Args:
        coeffs: (3, 250) coefficient array
        points_real: (N, 10) array of sample points
        psi: Complex quintic parameter
        k: Number of points to refine
        n_refine_steps: Newton iterations
        metric: 'FS' or 'k4_fermat'

    Returns:
        Fitness value (0 if Newton's method fails to converge)
    """
    min_set_real, _, newton_check_pass = filter_and_refine(
        points_real, coeffs, psi, k, n_refine_steps, filter_newton=True,
        dist_chunk_size=DIST_CHUNK_SIZE
    )
    fitness = jax.lax.cond(
        newton_check_pass,
        lambda points: compute_combined_fitness(
            min_set_real, coeffs, psi, metric
        ),
        lambda points: jnp.float32(0.0),
        min_set_real
    )

    return fitness

# -----------------------------------------------------------------------------
# 3. SPECIES MANAGEMENT & NICHING 
# -----------------------------------------------------------------------------
@jit
def calculate_distance(ind1, ind2):
    return jnp.linalg.norm(ind1.ravel() - ind2.ravel())

# --- NEW: Species Class ---
class Species:
    _id_counter = 0
    def __init__(self, representative):
        self.id = Species._id_counter
        Species._id_counter += 1
        self.representative = representative
        self.members = []
        self.fitness_values = []
        self.best_fitness = -jnp.inf
        self.generations_since_improvement = 0
        self.sigma = SIGMA_INIT
        self.sigma_cooldown = 0

    def add_member(self, individual, fitness):
        self.members.append(individual)
        self.fitness_values.append(fitness)

    def update_stagnation(self):
        current_max_fitness = jnp.max(jnp.array(self.fitness_values)) if self.members else -jnp.inf
        if current_max_fitness > self.best_fitness:
            self.best_fitness = current_max_fitness
            improved = True
            self.generations_since_improvement = 0
        else:
            improved = False
            self.generations_since_improvement += 1

        # Adaptive step size with cooldown
        if self.sigma_cooldown > 0:
            self.sigma_cooldown -= 1
        else:
            if improved:
                self.sigma = min(self.sigma * SIGMA_INCREASE, SIGMA_MAX)
            else:
                self.sigma = max(self.sigma * SIGMA_DECAY, SIGMA_MIN)
            self.sigma_cooldown = SIGMA_COOLDOWN

    def clear_members(self):
        self.members = []
        self.fitness_values = []

    def __repr__(self):
        return f"Species(id={self.id}, members={len(self.members)}, best_fitness={self.best_fitness:.4f}, stagnated_for={self.generations_since_improvement}, sigma={self.sigma:.3f})"

# -----------------------------------------------------------------------------
# 4. GENETIC ALGORITHM OPERATORS (REFACTORED FOR SPECIES)
# -----------------------------------------------------------------------------
@partial(jit, static_argnames=('k',))
def tournament_selection(key, population, fitness, k):
    num_individuals = population.shape[0]
    indices = jax.random.randint(key, (k,), 0, num_individuals)
    participants_fitness = fitness[indices]
    winner_index_in_tournament = jnp.argmax(participants_fitness)
    return population[indices[winner_index_in_tournament]]

# SBX Crossover and Polynomial Mutation remain the same as they operate on individuals.
# I'm including them here for completeness but they are unchanged from your original code.
@partial(jit, static_argnames=('eta',))
def sbx_crossover(key, parent1, parent2, eta):
    u = jax.random.uniform(key, shape=parent1.shape)
    beta = jnp.where(u <= 0.5, (2 * u)**(1 / (eta + 1)), (1 / (2 * (1 - u)))**(1 / (eta + 1)))
    offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    offspring1 = jnp.clip(offspring1, -1.0, 1.0)
    offspring2 = jnp.clip(offspring2, -1.0, 1.0)
    return offspring1, offspring2

@partial(jit, static_argnames=('prob_mut', 'eta'))
def polynomial_mutation(key, individual, prob_mut, eta, sigma=1.0):
    key1, key2 = jax.random.split(key)
    u = jax.random.uniform(key1, shape=individual.shape)
    do_mutation = jax.random.uniform(key2, shape=individual.shape) < prob_mut
    delta = jnp.where(u < 0.5, (2 * u)**(1 / (eta + 1)) - 1, 1 - (2 * (1 - u))**(1 / (eta + 1)))
    mutated_individual = jnp.where(do_mutation, individual + sigma * delta, individual)
    return jnp.clip(mutated_individual, -1.0, 1.0)

@partial(jit, static_argnames=('max_offspring', 'k_tournament', 'p_mut', 'eta_cross', 'eta_mut'))
def generate_padded_offspring_batch_with_crossover(key, members, fitness, max_offspring, k_tournament, p_mut, eta_cross, eta_mut, sigma=1.0):
    """Generates a fixed-size batch of offspring and we slice from it later."""

    # Create 5 master keys: one per stochastic op plus a dedicated key for the
    # keep-crossover-result decision (must be independent of cross_keys, which
    # parameterize the crossover itself).
    key_p1, key_p2, key_cross, key_mask, key_mut = jax.random.split(key, 5)

    # Split each master key into a batch of size max_offspring
    p1_keys = jax.random.split(key_p1, max_offspring)
    p2_keys = jax.random.split(key_p2, max_offspring)
    cross_keys = jax.random.split(key_cross, max_offspring)
    mask_keys = jax.random.split(key_mask, max_offspring)
    mut_keys = jax.random.split(key_mut, max_offspring)

    # VMAP to perform batched tournament selection
    select_fn = partial(tournament_selection, population=members, fitness=fitness, k=k_tournament)
    parent1_batch = vmap(select_fn)(p1_keys)
    parent2_batch = vmap(select_fn)(p2_keys)

    # VMAP to perform batched crossover
    crossover_fn = partial(sbx_crossover, eta=eta_cross)
    offspring1_batch, offspring2_batch = vmap(crossover_fn)(cross_keys, parent1_batch, parent2_batch)

    # Decide which children to keep based on crossover rate
    crossover_mask = vmap(jax.random.uniform)(mask_keys).reshape(-1, 1, 1) < CROSSOVER_RATE
    child_batch = jnp.where(crossover_mask, offspring1_batch, parent1_batch)

    # VMAP to perform batched mutation with adaptive step size
    mutation_fn = partial(polynomial_mutation, prob_mut=p_mut, eta=eta_mut, sigma=sigma)
    mutated_batch = vmap(mutation_fn)(mut_keys, child_batch)

    # VMAP to normalize the final batch
    normalized_batch = vmap(lambda p: normalize_coeffs(canonicalize_coeffs(p)))(mutated_batch)

    return normalized_batch


@partial(jit, static_argnames=('max_offspring', 'k_tournament', 'p_mut', 'eta_mut'))
def generate_padded_offspring_batch(key, members, fitness, max_offspring, k_tournament, p_mut, eta_mut, sigma=1.0):
    """Generates a fixed-size batch of offspring using mutation only (no crossover)."""
    
    # Create 2 master keys: one for selection, one for mutation
    key_sel, key_mut = jax.random.split(key, 2)
    
    # Split each master key into a batch of size max_offspring
    sel_keys = jax.random.split(key_sel, max_offspring)
    mut_keys = jax.random.split(key_mut, max_offspring)
    
    # VMAP to perform batched tournament selection
    select_fn = partial(tournament_selection, population=members, fitness=fitness, k=k_tournament)
    parent_batch = vmap(select_fn)(sel_keys)
    
    # VMAP to perform batched mutation
    mutation_fn = partial(polynomial_mutation, prob_mut=p_mut, eta=eta_mut, sigma=sigma)
    mutated_batch = vmap(mutation_fn)(mut_keys, parent_batch)
    
    # VMAP to normalize the final batch
    normalized_batch = vmap(lambda p: normalize_coeffs(canonicalize_coeffs(p)))(mutated_batch)
    
    return normalized_batch


def reproduce_within_species(key, species, num_offspring, tournament_size, eta_mutation, eta_crossover, mutation_rate, sigma=1.0):
    """Generates offspring by calling a padded, JIT-compiled function."""
    if num_offspring <= 0:
        return []

    members_arr = jnp.array(species.members)
    fitness_arr = jnp.array(species.fitness_values)
    num_members = members_arr.shape[0]

    # Elitism is handled here
    elite_offspring = []
    if SPECIES_ELITISM > 0 and num_members > 0:
        num_elites = min(SPECIES_ELITISM, num_offspring)
        elite_indices = jnp.argsort(fitness_arr)[-num_elites:]
        elite_offspring = [members_arr[i] for i in elite_indices]

    offspring_to_generate = num_offspring - len(elite_offspring)
    if offspring_to_generate <= 0:
        return elite_offspring

    MAX_OFFSPRING_PER_SPECIES = 128

    if num_members < 2:
        mutation_fn = partial(polynomial_mutation, prob_mut=mutation_rate, eta=eta_mutation, sigma=sigma)
        padded_offspring = vmap(mutation_fn, in_axes=(0, 0))(
            jax.random.split(key, offspring_to_generate),
            jnp.tile(members_arr, (offspring_to_generate, 1, 1))
        )
    else:
        MAX_SPECIES_SIZE = POPULATION_SIZE
        padding_size = MAX_SPECIES_SIZE - num_members
        padded_members = jnp.concatenate([
            members_arr,
            jnp.tile(members_arr[0:1], (padding_size, 1, 1))
        ])
        padded_fitness = jnp.concatenate([
            fitness_arr,
            jnp.full(padding_size, -jnp.inf)
        ])

        # Pass the dynamic parameters to the JIT-compiled function
        padded_offspring = generate_padded_offspring_batch(
            key, padded_members, padded_fitness, MAX_OFFSPRING_PER_SPECIES,
            tournament_size, mutation_rate, eta_mutation, sigma
        )

    new_offspring = padded_offspring[:offspring_to_generate]
    final_offspring = vmap(lambda p: normalize_coeffs(canonicalize_coeffs(p)))(new_offspring)

    return elite_offspring + list(final_offspring)


# 5. MAIN GA LOOP (WITH FULL CHECKPOINTING)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a Speciation-based Genetic Algorithm.")
    parser.add_argument(
        '--load_checkpoint', type=str, nargs='?', const='latest', default=None,
        help="Load a checkpoint. Use 'latest' to load the most recent, or provide a filename."
    )
    parser.add_argument(
        '--job_id', type=str, nargs='?', const='0', default='0',
        help="Provide a label, e.g. slurm job id, for the current run."
    )
    parser.add_argument(
        '--preload_d1', action='store_true',
        help="Preload the initial population with a good approximation from the d=1 case."
    )
    args = parser.parse_args()
    print("--- Speciation-based GA with Adaptive Schedule ---")
    print(f"Population: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}")
    print(f"Switching to exploitation mode at generation {TRANSITION_GENERATION}")


    # --- Load points ---
    print(f"Loading points from {POINTS_FILE}")
    points_real = load_points(POINTS_FILE)

    # --- Create the vmapped/pmapped fitness function ---
    num_devices = jax.local_device_count()
    print(f"Detected {num_devices} GPU(s).")

    vmap_fitness_batch = vmap(
        calculate_fitness_for_one_individual,
        in_axes=(0, None, None, None, None, None), out_axes=0
    )

    if num_devices > 1:
        evaluate_fitness = jax.pmap(
            vmap_fitness_batch,
            in_axes=(0, None, None, None, None, None),
            static_broadcasted_argnums=(3, 4, 5)
        )
    else:
        evaluate_fitness = vmap_fitness_batch

    # --- Base individual from d=1 coefficients ---
    d1_coeffs = jnp.array([
        [-0.2085878998041153, 0.08078225702047348, 0.12364989519119263, 0.42693421244621277, -0.4276507794857025, 0.05941963940858841, -0.19358153641223907, 0.2884068787097931, 0.2374262660741806, 0.17124612629413605, -0.03099866583943367, 0.07415380328893661, -0.22672683000564575, -0.1914607286453247, 0.09337177127599716, -0.053066715598106384, -0.06608302891254425, -0.3771730363368988, 0.05378381162881851, 0.0064529310911893845, 0.2938925623893738, 0.08852922171354294, 0.020463770255446434, 0.09666207432746887, -0.006990742404013872],
        [-0.1065014973282814, 0.20087268948554993, 0.18935158848762512, -0.17352613806724548, 0.05884088575839996, -0.4646260440349579, -0.10628655552864075, -0.28338274359703064, -0.03379037603735924, 0.007989203557372093, -0.06132059171795845, -0.13810740411281586, 0.04504100978374481, 0.015115765854716301, -0.4030528962612152, -0.025872472673654556, -0.4061300754547119, -0.02022559940814972, -0.13893099129199982, 0.10193423181772232, 0.29334160685539246, 0.22542181611061096, -0.050897762179374695, 0.21366965770721436, -0.04277477413415909],
        [0.054688308387994766, 0.07500440627336502, 0.060474496334791183, -0.3848169445991516, -0.3781052529811859, 0.38639041781425476, 0.021527282893657684, 0.4060642719268799, -0.15761728584766388, -0.1271764189004898, -0.01066557876765728, -0.13985656201839447, 0.1605837494134903, 0.15716029703617096, -0.32516127824783325, 0.016290534287691116, 0.2249980866909027, -0.2878168523311615, -0.12032820284366608, -0.04713383689522743, 0.025025269016623497, 0.08448748290538788, 0.05337755009531975, 0.05431513488292694, -0.03361976519227028]
    ])
    base_d1_individual = jnp.zeros(GENOTYPE_SHAPE)
    base_d1_individual = base_d1_individual.at[:, :25].set(d1_coeffs)

    # --- Load from checkpoint or initialize ---
    start_gen = 0
    cooldown_timer = COOLDOWN_GENERATIONS
    population = None
    species_list = []
    current_speciation_threshold = SPECIATION_THRESHOLD_INIT
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    checkpoint_to_load = None
    if args.load_checkpoint == 'latest':
        checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, 'checkpoint_gen_*.pkl'))
        if checkpoint_files:
            latest_gen = -1
            for f in checkpoint_files:
                match = re.search(r'checkpoint_gen_(\d+).pkl', f)
                if match:
                    gen_num = int(match.group(1))
                    if gen_num > latest_gen:
                        latest_gen = gen_num
                        checkpoint_to_load = f
    elif args.load_checkpoint is not None:
        checkpoint_to_load = os.path.join(CHECKPOINT_DIR, args.load_checkpoint)

    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        print(f"\nCheckpoint file found. Resuming run from: {checkpoint_to_load}")
        with open(checkpoint_to_load, 'rb') as f:
            checkpoint = pickle.load(f)
        population = checkpoint['population']
        start_gen = checkpoint['generation'] + 1
        key = checkpoint['key']
        species_list = checkpoint['species_list']
        current_speciation_threshold = checkpoint.get('speciation_threshold', SPECIATION_THRESHOLD_INIT)
        population = jnp.asarray(population) # Ensure it's a JAX array

        # IMPORTANT: Reset the species ID counter to avoid collisions
        if species_list:
            max_id = max(s.id for s in species_list)
            Species._id_counter = max_id + 1
        
    else:
        if args.load_checkpoint:
            print(f"Warning: Checkpoint '{args.load_checkpoint}' not found.")
        print("\nNo valid checkpoint specified. Starting a new run.")
        key, subkey = jax.random.split(key)
        
        if args.preload_d1:
            print("Preloading population: 25% near d=1 baseline, 75% with wide d=2 perturbation...")
            n_pure = POPULATION_SIZE // 4
            n_wide = POPULATION_SIZE - n_pure
            k_pure, k_wide = jax.random.split(subkey)
            noise_pure = jax.random.uniform(k_pure, (n_pure, *GENOTYPE_SHAPE), minval=-0.01, maxval=0.01)
            d2_wide = jax.random.uniform(k_wide, (n_wide, GENOTYPE_SHAPE[0], 225), minval=-0.2, maxval=0.2)
            d1_wide = jnp.zeros((n_wide, GENOTYPE_SHAPE[0], 25))
            noise_wide = jnp.concatenate([d1_wide, d2_wide], axis=2)
            population = jnp.concatenate([
                base_d1_individual + noise_pure,
                base_d1_individual + noise_wide,
            ], axis=0)
        else:
            population = jax.random.uniform(subkey, (POPULATION_SIZE, *GENOTYPE_SHAPE), minval=-1.0, maxval=1.0)
            
        population = vmap(lambda p: normalize_coeffs(canonicalize_coeffs(p)))(population)
        species_list = []
    
    print(f"\nStarting evolution from generation {start_gen}...")
    start_time = time.time()
    last_log_time = start_time # Initialize timer for logging intervals
    
    # --- Main Evolution Loop ---
    end_gen = start_gen + NUM_GENERATIONS
    for gen in range(start_gen, end_gen):
        
        # --- Set parameters based on the current generation ---
        if gen < TRANSITION_GENERATION:
            current_tourney_size = TOURNEY_SIZE_EXPLORE
            current_eta_mutation = ETA_MUTATION_EXPLORE
            current_eta_crossover = ETA_CROSSOVER_EXPLORE
            current_mutation_rate = MUTATION_RATE_EXPLORE
            territory_buffer = TERRITORY_BUFFER_EXPLORE
            
        else:
            current_tourney_size = TOURNEY_SIZE_EXPLOIT
            current_eta_mutation = ETA_MUTATION_EXPLOIT
            current_eta_crossover = ETA_CROSSOVER_EXPLOIT
            current_mutation_rate = MUTATION_RATE_EXPLOIT
            territory_buffer = TERRITORY_BUFFER_EXPLOIT
            
        # 1. Calculate fitness for the entire population
        all_fitness_scores = jnp.zeros(POPULATION_SIZE)
        num_batches = (POPULATION_SIZE + FITNESS_MINI_BATCH_SIZE - 1) // FITNESS_MINI_BATCH_SIZE
        
        for i in range(num_batches):
            start_idx = i * FITNESS_MINI_BATCH_SIZE
            end_idx = min(start_idx + FITNESS_MINI_BATCH_SIZE, POPULATION_SIZE)
            pop_batch = population[start_idx:end_idx]
            if num_devices > 1:
                per_device = pop_batch.shape[0] // num_devices
                pop_shards = [pop_batch[d * per_device:(d + 1) * per_device] for d in range(num_devices)]
                pop_sharded = device_put_sharded(pop_shards, jax.local_devices())
                fitness_reshaped = evaluate_fitness(
                    pop_sharded, points_real, PSI, MINSET_SIZE, NEWTON_STEPS, METRIC
                )
                fitness_batch = fitness_reshaped.reshape(-1)
            else:
                fitness_batch = evaluate_fitness(
                    pop_batch, points_real, PSI, MINSET_SIZE, NEWTON_STEPS, METRIC
                )

            # Replace any potential NaN/inf values with 0 before storing them.
            safe_fitness_batch = jnp.nan_to_num(fitness_batch, nan=0.0, posinf=0.0, neginf=0.0)
            all_fitness_scores = all_fitness_scores.at[start_idx:end_idx].set(safe_fitness_batch)

        # 2. Speciate the population
        for s in species_list: s.clear_members()
        
        if not species_list: # Handle first generation case
            new_species = Species(representative=population[0])
            species_list.append(new_species)

        try:
            # Merge close species
            species_list.sort(key=lambda s:s.best_fitness, reverse=True)
            representatives = jnp.array([s.representative for s in species_list])

            dist_matrix = calculate_distance_matrix(representatives, representatives)
            merge_threshold = current_speciation_threshold * SPECIATION_MERGE_RATIO
            is_close = dist_matrix < merge_threshold
            # Always merge the lower fitness species to the higher ones
            keep_mask = ~jnp.any(jnp.tril(is_close, k=-1), axis=1)

            keep_mask_np = np.array(keep_mask) # Convert to CPU
            species_list = [s for i, s in enumerate(species_list) if keep_mask_np[i]]

        except Exception as e:
            print(f"Warning: species merge failed at gen {gen+1}: {e}")

        # Create a matrix of all species representatives
        representatives = jnp.array([s.representative for s in species_list])
        dist_matrix = calculate_distance_matrix(population, representatives)
        
        # Find the closest species index for each individual in one go
        closest_species_indices = jnp.argmin(dist_matrix, axis=1)
        
        # Check for new species
        min_distances = jnp.min(dist_matrix, axis=1)
        new_species_mask = min_distances >=  current_speciation_threshold

        # This part must run on the CPU as it modifies Python objects
        for i in range(POPULATION_SIZE):
            if new_species_mask[i]:
                new_species = Species(representative=population[i])
                new_species.add_member(population[i], all_fitness_scores[i])
                species_list.append(new_species)
            else:
                species_idx = closest_species_indices[i]
                species_list[species_idx].add_member(population[i], all_fitness_scores[i])


        # --- Update representatives to prevent drift ---
        for s in species_list:
            if s.members:
                # Find the best member of the current generation
                best_member_idx = jnp.argmax(jnp.array(s.fitness_values))
                # Update the representative to this new best member
                s.representative = s.members[best_member_idx]


        # --- Dynamic Threshold Adjustment ---
        if gen > WARMUP_GENERATIONS: 
            if cooldown_timer > 0:
                cooldown_timer -= 1
            else:
                current_species_count = len(species_list)
                if current_species_count < TARGET_SPECIES_COUNT_MIN:
                    current_speciation_threshold *= 1 - SPECIATION_THRESHOLD_STEP
                    cooldown_timer = COOLDOWN_GENERATIONS
                elif current_species_count > TARGET_SPECIES_COUNT_MAX:
                    current_speciation_threshold *= 1 + SPECIATION_THRESHOLD_STEP
                    cooldown_timer = COOLDOWN_GENERATIONS


                # 3. Calculate offspring allocation
        # --- NEW: Species-Level Fitness Sharing ---
        
        # Step 1: Calculate the raw average fitness for each species
        raw_avg_fitness = jnp.array([jnp.mean(jnp.array(s.fitness_values)) if s.members else 0.0 for s in species_list])
        
        # Step 2: Calculate distances between all species representatives
        # We already have the 'representatives' array from the speciation step.
        representatives = jnp.array([s.representative for s in species_list])
        species_dist_matrix = calculate_distance_matrix(representatives, representatives)

        current_territory_radius = current_speciation_threshold + territory_buffer

        # Step 3: Calculate niche crowding for each species
        # A species is "crowded" by another if the distance is < the sharing radius.
        # We get a boolean matrix of shape (num_species, num_species).
        sharing_matrix = species_dist_matrix < current_territory_radius
        
        # The niche count is the sum of True values in each row.
        niche_counts = jnp.sum(sharing_matrix, axis=1)
        # Ensure niche_count is at least 1 to avoid division by zero.
        niche_counts = jnp.maximum(1.0, niche_counts)

        # Step 4: Adjust fitness by dividing by the crowding count.
        # This heavily penalizes species that are in crowded regions.
        adjusted_fitness = raw_avg_fitness / niche_counts
        
        final_adjusted_fitness = jnp.maximum(0, adjusted_fitness) # Clamp for safety
        total_adjusted_fitness = jnp.sum(final_adjusted_fitness)
        
        next_generation_population = []
        if total_adjusted_fitness > 0:
            proportions = (final_adjusted_fitness / total_adjusted_fitness) * POPULATION_SIZE
            offspring_counts = jnp.round(proportions).astype(int)
            diff = POPULATION_SIZE - jnp.sum(offspring_counts)
            if diff != 0:
                idx_to_update = jnp.argmax(final_adjusted_fitness)
                offspring_counts = offspring_counts.at[idx_to_update].set(offspring_counts[idx_to_update] + diff)
            
            # The reproduction loop now uses the adaptive parameters
            for i, s in enumerate(species_list):
                if s.members:
                    num_offspring = int(offspring_counts[i])
                    key, subkey = jax.random.split(key)
                    offspring = reproduce_within_species(
                        subkey, s, num_offspring,
                        current_tourney_size, current_eta_mutation, current_eta_crossover, current_mutation_rate,
                        sigma=s.sigma
                    )
                    next_generation_population.extend(offspring)

        # 4. Handle stagnation and create new population
        for s in species_list: s.update_stagnation()
        
        # Prune stale species, but keep at least one
        survival_threshold = jnp.max(all_fitness_scores) * STAGNATION_SURVIVAL_RATIO
        species_list = [s for s in species_list if (
                                s.generations_since_improvement < STAGNATION_THRESHOLD or s.best_fitness > survival_threshold)
                                and s.members]

        # Ensure population size is maintained
        if len(next_generation_population) != POPULATION_SIZE:
             # This can happen due to rounding or empty species. Refill if necessary.
             current_pop_size = len(next_generation_population)
             if current_pop_size < POPULATION_SIZE:
                 key, subkey = jax.random.split(key)
                 randoms_needed = POPULATION_SIZE - current_pop_size
                 if args.preload_d1:
                     n_pure = randoms_needed // 4
                     n_wide = randoms_needed - n_pure
                     k_pure, k_wide = jax.random.split(subkey)
                     noise_pure = jax.random.uniform(k_pure, (n_pure, *GENOTYPE_SHAPE), minval=-0.01, maxval=0.01)
                     d2_wide = jax.random.uniform(k_wide, (n_wide, GENOTYPE_SHAPE[0], 225), minval=-0.2, maxval=0.2)
                     d1_wide = jnp.zeros((n_wide, GENOTYPE_SHAPE[0], 25))
                     noise_wide = jnp.concatenate([d1_wide, d2_wide], axis=2)
                     random_individuals = jnp.concatenate([
                         base_d1_individual + noise_pure,
                         base_d1_individual + noise_wide,
                     ], axis=0)
                 else:
                     random_individuals = jax.random.uniform(subkey, (randoms_needed, *GENOTYPE_SHAPE), minval=-1.0, maxval=1.0)
                 random_individuals = vmap(lambda p: normalize_coeffs(canonicalize_coeffs(p)))(random_individuals)
                 next_generation_population.extend(random_individuals)

        population = jnp.stack(next_generation_population[:POPULATION_SIZE])
        
        # 5. Logging
        if (gen + 1) % LOG_INTERVAL == 0:
            current_time = time.time()
            duration_for_interval = current_time - last_log_time
            avg_time_per_gen = duration_for_interval / LOG_INTERVAL

            max_fitness = jnp.max(all_fitness_scores)
            avg_fitness = jnp.mean(all_fitness_scores)
            sigmas = [s.sigma for s in species_list if s.members]
            sigma_str = f"Sigma: {min(sigmas):.2f}/{max(sigmas):.2f}" if sigmas else "Sigma: -"
            print(f"Gen {gen+1:4d}/{end_gen} | Species: {len(species_list):2d} | Threshold: {current_speciation_threshold:.2f} | {sigma_str} | Max Fit: {max_fitness:.4f} | Avg Fit: {avg_fitness:.4f} | Avg Gen Time: {avg_time_per_gen:.2f}s")
        
            last_log_time = current_time # Reset timer for the next interval
        # 6. Checkpointing
        if (gen + 1) % CHECKPOINT_INTERVAL == 0 and (gen + 1) <= end_gen:
            checkpoint_filename = os.path.join(CHECKPOINT_DIR, f'checkpoint_gen_{gen+1}.pkl')
            # Prune members from species before saving to reduce file size
            # The representative and stagnation state is the important part
            species_to_save = [Species(s.representative) for s in species_list]
            for i, s in enumerate(species_to_save):
                s.id = species_list[i].id
                s.best_fitness = species_list[i].best_fitness
                s.generations_since_improvement = species_list[i].generations_since_improvement
                s.sigma = species_list[i].sigma
                s.sigma_cooldown = species_list[i].sigma_cooldown

            checkpoint_data = {
                'population': population,
                'generation': gen,
                'key': key,
                'species_list': species_to_save,
                'speciation_threshold': current_speciation_threshold
            }
            with open(checkpoint_filename, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"--- Checkpoint saved to {checkpoint_filename} ---")


    end_time = time.time()
    print(f"\nEvolution finished in {end_time - start_time:.2f} seconds.")


    # --- Final Analysis ---
    print("\n--- Analyzing Final Population ---")

    # 1. Re-calculate fitness for the FINAL population to ensure it's up-to-date.
    print("Calculating final fitness scores...")
    final_fitness = jnp.zeros(POPULATION_SIZE)
    num_batches = (POPULATION_SIZE + FITNESS_MINI_BATCH_SIZE - 1) // FITNESS_MINI_BATCH_SIZE
    for i in range(num_batches):
        start_idx = i * FITNESS_MINI_BATCH_SIZE
        end_idx = min(start_idx + FITNESS_MINI_BATCH_SIZE, POPULATION_SIZE)
        population_batch = population[start_idx:end_idx]
        if num_devices > 1:
            per_device = population_batch.shape[0] // num_devices
            pop_shards = [population_batch[d * per_device:(d + 1) * per_device] for d in range(num_devices)]
            pop_sharded = device_put_sharded(pop_shards, jax.local_devices())
            fitness_reshaped = evaluate_fitness(
                pop_sharded, points_real, PSI, MINSET_SIZE, NEWTON_STEPS, METRIC
            )
            fitness_batch = fitness_reshaped.reshape(-1)
        else:
            fitness_batch = evaluate_fitness(
                population_batch, points_real, PSI, MINSET_SIZE, NEWTON_STEPS, METRIC
            )
        safe_fitness_batch = jnp.nan_to_num(fitness_batch, nan=0.0, posinf=0.0, neginf=0.0)
        final_fitness = final_fitness.at[start_idx:end_idx].set(safe_fitness_batch)

    # 2. Use the correct vectorized speciation to assign members.
    for s in species_list: s.clear_members()
    
    if species_list:
        representatives = jnp.array([s.representative for s in species_list])
        dist_matrix = calculate_distance_matrix(population, representatives)
        
        closest_species_indices = jnp.argmin(dist_matrix, axis=1)

        for i in range(POPULATION_SIZE):
            species_idx = closest_species_indices[i]
            # Ensure the species still exists before trying to add a member
            if species_idx < len(species_list):
                 species_list[species_idx].add_member(population[i], final_fitness[i])

    # Filter out any species that are now empty after re-assignment
    final_species_list = [s for s in species_list if s.members]
    
    print(f"\nFound {len(final_species_list)} distinct species with members in the final population.")

    # Sort the populated species by their best current fitness
    final_species_list.sort(key=lambda s: jnp.max(jnp.array(s.fitness_values)), reverse=True)

    if final_species_list:
        top_fitness_score = jnp.max(jnp.array(final_species_list[0].fitness_values))
    else:
        top_fitness_score = 0.0

    # Per-species run folders for species with best_fitness >= 0.5 * top_fitness_score.
    # run_fitness_pipeline writes coeffs.pkl as part of the sidecar contract,
    # so each species's folder is self-contained (coeffs + sidecars + plots).
    # Species list is fitness-descending, so we break at the first below-threshold.
    rank = 1
    for s in final_species_list:
        best_member_idx = jnp.argmax(jnp.array(s.fitness_values))
        best_member = s.members[best_member_idx]
        best_fitness = s.fitness_values[best_member_idx]

        if best_fitness < 0.5 * top_fitness_score:
            print(f"\nStopping plot generation: Species {s.id} fitness ({best_fitness:.5f}) is below half of the top fitness ({top_fitness_score:.5f}).")
            break

        print(f"\n--- Species {s.id} (Best Fitness: {best_fitness:.5f}) ---")
        print(f"Size: {len(s.members)} members | Stagnated for: {s.generations_since_improvement} gens")
        print("Best Member's Coefficients:")
        print(format_array_with_commas(best_member))

        out_dir = os.path.join(
            f'plots_slag_{args.job_id}',
            f'plots_slag_{args.job_id}_{rank}_id{s.id}'
        )
        run_fitness_pipeline(points_real, best_member, PSI, k=100000, n_refine_steps=100, metric=METRIC, compare_with="random", out_dir=out_dir)
        rank += 1
