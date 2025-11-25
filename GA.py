import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from functools import partial
import pickle
import time
import os
import argparse
import glob
import re
import sys

# --- User Custom Modules ---
from find_smooth_submanifold import filter_and_refine
from slag_condition import compute_combined_fitness
from helper import format_array_with_commas
from plots import make_fitness_plots

jax.config.update('jax_default_matmul_precision', 'highest')

# -----------------------------------------------------------------------------
# 1. HYPERPARAMETERS
# -----------------------------------------------------------------------------
PSI = 0
CYPOINTSFILE = f'/projects/ruehlehet/yidi/sLag/data_psi/1mil_patch_all_psi{PSI}_seed1024.pkl'

# Metric used when compute the kahler form
# Options are 1. FS - Fubini-Study metric
#             2. k4_fermat - Ricci-flat metric with k = 4 in Donaldson's construction. Fermat only.
# METRIC = 'FS'
METRIC = 'k4_fermat'

# GA Parameters
POPULATION_SIZE = 600
GENOTYPE_SHAPE = (293,)
NUM_GENES = GENOTYPE_SHAPE[0]
NUM_GENERATIONS = 400
TRANSITION_GENERATION = 400

# --- WEIGHT BOUNDS (Expanded for tanh saturation) ---
WEIGHT_BOUND_MIN = -20.0
WEIGHT_BOUND_MAX = 20.0

# --- Dynamic Speciation Parameters ---
TARGET_SPECIES_COUNT = 15
SPECIATION_THRESHOLD_INIT = 3.0
MIN_SPECIATION_THRESHOLD = 0.1
SPECIATION_THRESHOLD_STEP = 0.5

# --- Inter-Species Repulsion ("Territory") ---
TERRITORY_BUFFER_EXPLORE = 1.5
TERRITORY_BUFFER_EXPLOIT = 0.5

# --- Exploration Phase Settings ---
TOURNEY_SIZE_EXPLORE = 3
MUTATION_RATE_EXPLORE = 2.5 / NUM_GENES 
# Higher Eta (30) prevents "blowout" on the large [-20, 20] range
ETA_MUTATION_EXPLORE = 30.0
ETA_CROSSOVER_EXPLORE = 15.0

# --- Exploitation Phase Settings ---
TOURNEY_SIZE_EXPLOIT = 7
MUTATION_RATE_EXPLOIT = 0.5 / NUM_GENES 
# Very high Eta (300) forces tiny, local-search steps for Newton convergence
ETA_MUTATION_EXPLOIT = 300.0  
ETA_CROSSOVER_EXPLOIT = 30.0

# General Operator Params
CROSSOVER_RATE = 0.9
STAGNATION_THRESHOLD = 20 # Generations a species can go without improvement before being removed.     
SPECIES_ELITISM = 1       # Number of best individuals per species to carry over directly.  

# Batching & System
FITNESS_MINI_BATCH_SIZE = 25
LOG_INTERVAL = 1
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_INTERVAL = 100
MINSET_SIZE = 10000
NEWTON_STEPS = 40

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
        coeffs: (3, 25) coefficient
        points_real: (N, 10) array of sample points
        psi: Complex quintic parameter
        k: Number of points to refine
        n_refine_steps: Newton iterations
        metric: 'FS' or 'k4_fermat'

    Returns:
        Fitness value (0 if Newton's method fails to converge)
    """

    min_set_real, _, newton_check_pass = filter_and_refine(
        points_real, coeffs, psi, k, n_refine_steps, filter_newton=True
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
    return jnp.mean(jnp.abs(ind1.ravel() - ind2.ravel()))

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

    def add_member(self, individual, fitness):
        self.members.append(individual)
        self.fitness_values.append(fitness)

    def update_stagnation(self):
        current_max_fitness = jnp.max(jnp.array(self.fitness_values)) if self.members else -jnp.inf
        if current_max_fitness > self.best_fitness:
            self.best_fitness = current_max_fitness
            self.generations_since_improvement = 0
        else:
            self.generations_since_improvement += 1

    def clear_members(self):
        self.members = []
        self.fitness_values = []

    def __repr__(self):
        return f"Species(id={self.id}, size={len(self.members)}, best_fitness={self.best_fitness:.4f},  stagnated_for={self.generations_since_improvement})"

# -----------------------------------------------------------------------------
# 4. GENETIC ALGORITHM OPERATORS (FIXED SCALING)
# -----------------------------------------------------------------------------
@partial(jit, static_argnames=('k',))
def tournament_selection(key, population, fitness, k):
    num_individuals = population.shape[0]
    indices = jax.random.randint(key, (k,), 0, num_individuals)
    participants_fitness = fitness[indices]
    winner_index_in_tournament = jnp.argmax(participants_fitness)
    return population[indices[winner_index_in_tournament]]

@partial(jit, static_argnames=('eta', 'min_val', 'max_val'))
def sbx_crossover(key, parent1, parent2, eta, min_val, max_val):
    u = jax.random.uniform(key, shape=parent1.shape)
    beta = jnp.where(u <= 0.5, (2 * u)**(1 / (eta + 1)), (1 / (2 * (1 - u)))**(1 / (eta + 1)))
    
    offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    
    # Clip to specific bounds
    offspring1 = jnp.clip(offspring1, min_val, max_val)
    offspring2 = jnp.clip(offspring2, min_val, max_val)
    return offspring1, offspring2

@partial(jit, static_argnames=('prob_mut', 'eta', 'min_val', 'max_val'))
def polynomial_mutation(key, individual, prob_mut, eta, min_val, max_val):
    key1, key2 = jax.random.split(key)
    u = jax.random.uniform(key1, shape=individual.shape)
    do_mutation = jax.random.uniform(key2, shape=individual.shape) < prob_mut
    
    # Calculate delta_q (Standard Deb logic)
    delta_q = jnp.where(u < 0.5, (2 * u)**(1 / (eta + 1)) - 1, 1 - (2 * (1 - u))**(1 / (eta + 1)))
    
    # --- FIX: Scale step size by the full range width ---
    range_width = max_val - min_val
    mutation_step = delta_q * range_width * 0.5 
    
    mutated_individual = jnp.where(do_mutation, individual + mutation_step, individual)
    
    # Clip to specific bounds
    return jnp.clip(mutated_individual, min_val, max_val)

@partial(jit, static_argnames=('max_offspring', 'k_tournament', 'p_mut', 'eta_cross', 'eta_mut', 'min_val', 'max_val'))
def generate_padded_offspring_batch(key, members, fitness, max_offspring, k_tournament, p_mut, eta_cross, eta_mut, min_val, max_val):
    """Generates a fixed-size batch of offspring and we slice from it later."""

    # Create 4 master keys, one for each stochastic operation.  
    key_p1, key_p2, key_cross, key_mut = jax.random.split(key, 4)
    p1_keys = jax.random.split(key_p1, max_offspring)
    p2_keys = jax.random.split(key_p2, max_offspring)
    cross_keys = jax.random.split(key_cross, max_offspring)
    mut_keys = jax.random.split(key_mut, max_offspring)

    # Batched tournament selection
    select_fn = partial(tournament_selection, population=members, fitness=fitness, k=k_tournament)
    parent1_batch = vmap(select_fn)(p1_keys)
    parent2_batch = vmap(select_fn)(p2_keys)

    # Crossover
    crossover_fn = partial(sbx_crossover, eta=eta_cross, min_val=min_val, max_val=max_val)
    offspring1_batch, offspring2_batch = vmap(crossover_fn)(cross_keys, parent1_batch, parent2_batch)

    # Decide which children to keep based on crossover rate 
    crossover_mask = vmap(jax.random.uniform)(cross_keys).reshape(-1, 1) < CROSSOVER_RATE
    child_batch = jnp.where(crossover_mask, offspring1_batch, parent1_batch)

    # Mutation
    mutation_fn = partial(polynomial_mutation, prob_mut=p_mut, eta=eta_mut, min_val=min_val, max_val=max_val)
    mutated_batch = vmap(mutation_fn)(mut_keys, child_batch)
    return mutated_batch

def reproduce_within_species(key, species, num_offspring, tournament_size, eta_mutation, eta_crossover, mutation_rate, min_val, max_val):
    """Generates offspring by calling a padded, JIT-compiled function.""" 
    if num_offspring <= 0:
        return []

    members_arr = jnp.array(species.members)
    fitness_arr = jnp.array(species.fitness_values)
    num_members = members_arr.shape[0]

    # Elitism
    elite_offspring = []
    if SPECIES_ELITISM > 0 and num_members > 0:
        num_elites = min(SPECIES_ELITISM, num_offspring)
        elite_indices = jnp.argsort(fitness_arr)[-num_elites:]
        elite_offspring = [members_arr[i] for i in elite_indices]
        
    offspring_to_generate = num_offspring - len(elite_offspring)
    if offspring_to_generate <= 0:
        return elite_offspring

    MAX_OFFSPRING_PER_SPECIES = 128 

    # If there is only one member, duplicate itself
    if num_members < 2:
        mutation_fn = partial(polynomial_mutation, prob_mut=mutation_rate, eta=eta_mutation, min_val=min_val, max_val=max_val)
        padded_offspring = vmap(mutation_fn, in_axes=(0, 0))(
            jax.random.split(key, offspring_to_generate), 
            jnp.tile(members_arr, (offspring_to_generate, 1))
        )
    # Otherwise, duplicate the first member and assign -inf to them just to hold the shape
    else:
        MAX_SPECIES_SIZE = POPULATION_SIZE 
        padding_size = MAX_SPECIES_SIZE - num_members
        padded_members = jnp.concatenate([members_arr, jnp.tile(members_arr[0:1], (padding_size, 1))])
        padded_fitness = jnp.concatenate([fitness_arr, jnp.full(padding_size, -jnp.inf)])
        
        padded_offspring = generate_padded_offspring_batch(
            key, padded_members, padded_fitness, MAX_OFFSPRING_PER_SPECIES,
            tournament_size, mutation_rate, eta_crossover, eta_mutation,
            min_val, max_val
        )
    
    new_offspring = padded_offspring[:offspring_to_generate]
    return elite_offspring + list(new_offspring)

# -----------------------------------------------------------------------------
# 5. MAIN GA LOOP
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a Speciation-based Genetic Algorithm.")
    parser.add_argument('--load_checkpoint', type=str, nargs='?', const='latest', default=None)
    parser.add_argument('--job_id', type=str, nargs='?', const='0', default='0')
    args = parser.parse_args()

    print("--- Speciation GA ---")
    print(f"Population: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}") 
    print(f"Switching to exploitation mode at generation {TRANSITION_GENERATION}")
    print(f"Weight Bounds: [{WEIGHT_BOUND_MIN}, {WEIGHT_BOUND_MAX}]")
    print(f"Params: Eta_Mut_Exp={ETA_MUTATION_EXPLORE}, Eta_Mut_Exploit={ETA_MUTATION_EXPLOIT}")

    # Load points
    with open(CYPOINTSFILE, 'rb') as f:
        points_real = np.asarray(pickle.load(f))
    points_real = np.concatenate([np.real(points_real), np.imag(points_real)], axis=1)
    points_real = jax.device_put(jnp.asarray(points_real))

    vmap_fitness_batch = vmap(
        calculate_fitness_for_one_individual,
        in_axes=(0, None, None, None, None, None), out_axes=0
    )

    # Initialization
    start_gen = 0
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
        population = jnp.asarray(checkpoint['population'])
        start_gen = checkpoint['generation'] + 1
        key = checkpoint['key']
        species_list = checkpoint['species_list']
        current_speciation_threshold = checkpoint.get('speciation_threshold', SPECIATION_THRESHOLD_INIT)

        if species_list:
            Species._id_counter = max(s.id for s in species_list) + 1
    else:
        if args.load_checkpoint: 
            print(f"Warning: Checkpoint '{args.load_checkpoint}' not found.") 
        print("\nNo valid checkpoint specified. Starting a new run.")

        key, subkey = jax.random.split(key)
        # WIDER INITIALIZATION: [-3, 3] to hit non-linearity early
        population = jax.random.uniform(subkey, (POPULATION_SIZE, *GENOTYPE_SHAPE), minval=-3.0, maxval=3.0)
        species_list = []
    
    print(f"\nStarting evolution from generation {start_gen}...")
    start_time = time.time()
    last_log_time = start_time 
   
    # --- Main Evolution Loop --- 
    end_gen = start_gen + NUM_GENERATIONS
    for gen in range(start_gen, end_gen):
        
        # --- Set Phase Parameters ---
        if gen < start_gen + TRANSITION_GENERATION:
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
            
        # 1. Fitness Evaluation
        all_fitness_scores = jnp.zeros(POPULATION_SIZE)
        num_batches = (POPULATION_SIZE + FITNESS_MINI_BATCH_SIZE - 1) // FITNESS_MINI_BATCH_SIZE
        for i in range(num_batches):
            start_idx = i * FITNESS_MINI_BATCH_SIZE
            end_idx = min(start_idx + FITNESS_MINI_BATCH_SIZE, POPULATION_SIZE)
            pop_batch = population[start_idx:end_idx]
            fitness_batch = vmap_fitness_batch(pop_batch, points_real, PSI, MINSET_SIZE, NEWTON_STEPS, METRIC)
            # Replace any potential NaN/inf values with 0 before storing them.
            safe_fitness_batch = jnp.nan_to_num(fitness_batch, nan=0.0, posinf=0.0, neginf=0.0)
            all_fitness_scores = all_fitness_scores.at[start_idx:end_idx].set(safe_fitness_batch)

        # 2. Speciation
        for s in species_list: s.clear_members()
        if not species_list: # Handle first generation case
            species_list.append(Species(representative=population[0]))

        # Create a matrix of all species representatives
        representatives = jnp.array([s.representative for s in species_list])

        dist_to_reps = vmap(calculate_distance, in_axes=(None, 0)) # ind vs all reps 
        dist_matrix = vmap(dist_to_reps, in_axes=(0, None))(population, representatives) # all inds vs all reps 
        
        closest_species_indices = jnp.argmin(dist_matrix, axis=1)

        # Check for new species
        min_distances = jnp.min(dist_matrix, axis=1)
        new_species_mask = min_distances >= current_speciation_threshold

        # This part must run on the CPU as it modifies Python objects
        for i in range(POPULATION_SIZE):
            if new_species_mask[i]:
                new_species = Species(representative=population[i])
                new_species.add_member(population[i], all_fitness_scores[i])
                species_list.append(new_species)
            else:
                idx = closest_species_indices[i]
                species_list[idx].add_member(population[i], all_fitness_scores[i])

        # --- Filter Empty Species ---
        species_list = [s for s in species_list if s.members]

        # --- Dynamic Threshold Adjustment ---
        current_species_count = len(species_list)
        if current_species_count < TARGET_SPECIES_COUNT:
            current_speciation_threshold = max(MIN_SPECIATION_THRESHOLD, current_speciation_threshold - SPECIATION_THRESHOLD_STEP)
        elif current_species_count > TARGET_SPECIES_COUNT:
            current_speciation_threshold += SPECIATION_THRESHOLD_STEP

        # Update Representatives
        for s in species_list:
            best_member_idx = jnp.argmax(jnp.array(s.fitness_values))
            s.representative = s.members[best_member_idx]

        # 3. Double Punishment: Size + Territory
        #    Fitness sharing within each species and across close species
        raw_avg_fitness = jnp.array([jnp.mean(jnp.array(s.fitness_values)) for s in species_list])
        representatives = jnp.array([s.representative for s in species_list])
        species_dist_matrix = vmap(vmap(calculate_distance, in_axes=(None, 0)), in_axes=(0, None))(representatives, representatives)

        current_territory_radius = current_speciation_threshold + territory_buffer

        sharing_matrix = species_dist_matrix < current_territory_radius
        niche_counts = jnp.sum(sharing_matrix, axis=1) 
        niche_counts = jnp.maximum(1.0, niche_counts)
        
        species_sizes = jnp.array([len(s.members) for s in species_list])
        total_crowding = species_sizes * niche_counts 
        
        adjusted_fitness = raw_avg_fitness / total_crowding
        final_adjusted_fitness = jnp.maximum(0, adjusted_fitness)

        # 4. Offspring Allocation
        total_fitness = jnp.sum(final_adjusted_fitness)
        next_generation_population = []
        
        if total_fitness > 0:
            proportions = (final_adjusted_fitness / total_fitness) * POPULATION_SIZE
            offspring_counts = jnp.round(proportions).astype(int)
            diff = POPULATION_SIZE - jnp.sum(offspring_counts)
            if diff != 0:
                idx = jnp.argmax(final_adjusted_fitness)
                offspring_counts = offspring_counts.at[idx].set(offspring_counts[idx] + diff)
            
            for i, s in enumerate(species_list):
                count = int(offspring_counts[i])
                if count > 0:
                    key, subkey = jax.random.split(key)
                    # Pass the bounds to reproduction
                    offspring = reproduce_within_species(
                        subkey, s, count,
                        current_tourney_size, current_eta_mutation, current_eta_crossover, current_mutation_rate,
                        WEIGHT_BOUND_MIN, WEIGHT_BOUND_MAX
                    )
                    next_generation_population.extend(offspring)
        
        # 5. Stagnation & Refill
        for s in species_list: s.update_stagnation()
       
        # Ensure population size is maintained  
        if len(next_generation_population) < POPULATION_SIZE:
            randoms_needed = POPULATION_SIZE - len(next_generation_population)
            key, subkey = jax.random.split(key)
            # Initialize refill with wider bounds
            random_individuals = jax.random.uniform(subkey, (randoms_needed, *GENOTYPE_SHAPE), minval=-3.0, maxval=3.0)
            next_generation_population.extend(random_individuals)

        population = jnp.array(next_generation_population[:POPULATION_SIZE])
        
        # 6. Logging
        if (gen + 1) % LOG_INTERVAL == 0:
            duration = time.time() - last_log_time
            avg_time_per_gen = duration / LOG_INTERVAL 
            max_fit = jnp.max(all_fitness_scores)
            avg_fit = jnp.mean(all_fitness_scores)
            print(f"Gen {gen+1:3d} | Sp: {len(species_list):2d} (Thr: {current_speciation_threshold:.2f}, Rad: {current_territory_radius:.2f}) | Max: {max_fit:.4f} | Avg: {avg_fit:.4f} | {avg_time_per_gen:.2f}s")
            last_log_time = time.time()

        # 7. Checkpointing
        if (gen + 1) % CHECKPOINT_INTERVAL == 0:
            fname = os.path.join(CHECKPOINT_DIR, f'checkpoint_gen_{gen+1}.pkl')
            save_species = [Species(s.representative) for s in species_list]
            for i, s in enumerate(save_species):
                s.id = species_list[i].id
                s.best_fitness = species_list[i].best_fitness
                s.generations_since_improvement = species_list[i].generations_since_improvement

            checkpoint_data = {
                'population': population, 'generation': gen, 'key': key, 
                'species_list': save_species, 
                'speciation_threshold': current_speciation_threshold
            }
            with open(fname, 'wb') as f: pickle.dump(checkpoint_data, f)
            print(f"--- Checkpoint saved to {fname} ---")

    print(f"Done in {time.time() - start_time:.2f}s")

    # --- FINAL ANALYSIS ---
    print("\n--- Analyzing Final Population ---")
    print("Calculating final fitness scores...")

    final_fitness = jnp.zeros(POPULATION_SIZE)
    num_batches = (POPULATION_SIZE + FITNESS_MINI_BATCH_SIZE - 1) // FITNESS_MINI_BATCH_SIZE

    # Recalculate fitness for the final population
    for i in range(num_batches):
        start_idx = i * FITNESS_MINI_BATCH_SIZE
        end_idx = jnp.minimum(start_idx + FITNESS_MINI_BATCH_SIZE, POPULATION_SIZE)
        population_batch = population[start_idx:end_idx]
        fitness_batch = vmap_fitness_batch(population_batch, points_real, PSI, MINSET_SIZE, NEWTON_STEPS, METRIC)
        final_fitness = final_fitness.at[start_idx:end_idx].set(fitness_batch)

    for s in species_list: s.clear_members()
    representatives = jnp.array([s.representative for s in species_list])
    dist_matrix = vmap(vmap(calculate_distance, in_axes=(None, 0)), in_axes=(0, None))(population, representatives)
    closest_species_indices = jnp.argmin(dist_matrix, axis=1)
    
    for i in range(POPULATION_SIZE):
        idx = closest_species_indices[i]
        if idx < len(species_list):
             species_list[idx].add_member(population[i], final_fitness[i])

    final_species_list = [s for s in species_list if s.members]
    print(f"\nFound {len(final_species_list)} distinct species with members in the final population.")
    final_species_list.sort(key=lambda s: jnp.max(jnp.array(s.fitness_values)), reverse=True)

    rank = 1
    for s in final_species_list:
        best_member_idx = jnp.argmax(jnp.array(s.fitness_values))
        best_member = s.members[best_member_idx]
        best_fitness = s.fitness_values[best_member_idx]

        print(f"\n--- Species {s.id} (Best Fitness: {best_fitness:.5f}) ---")
        print(f"Size: {len(s.members)} members | Stagnated for: {s.generations_since_improvement} gens")
        
        parent_folder = os.path.join(f'plots_slag_{args.job_id}', f'plots_slag_{args.job_id}_{rank}_id{s.id}')
        make_fitness_plots(points_real, best_member, PSI, k=10000, n_refine_steps=40, metric=METRIC, compare_with_random=False, parent_folder=parent_folder)
        rank += 1
