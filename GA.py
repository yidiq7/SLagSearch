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
from find_smooth_submanifold import filter_and_refine, normalize_coeffs, get_basis_labels, combine_to_complex_equations
from slag_condition import compute_combined_fitness
from helper import canonicalize_coeffs, format_array_with_commas
from plots import make_fitness_plots

jax.config.update('jax_default_matmul_precision', 'highest')

# -----------------------------------------------------------------------------
# 1. HYPERPARAMETERS
# -----------------------------------------------------------------------------
# Moduli of the quintic
#PSI = 1000000
#CYPOINTSFILE = f'/projects/ruehlehet/yidi/sLag/data_psi/5mil_patch0_psi{PSI}_seed1024.pkl'
PSI = 0
CYPOINTSFILE = f'/projects/ruehlehet/yidi/sLag/data_psi/1mil_patch_all_psi{PSI}_seed1024.pkl'
#CYPOINTSFILE = '/projects/ruehlehet/yidi/sLag/data/5mil_patch0_343.pkl'

# Metric used when compute the kahler form
# Options are 1. FS - Fubini-Study metric
#             2. k4_fermat - Ricci-flat metric with k = 4 in Donaldson's construction. Fermat only.
#METRIC = 'FS'
METRIC = 'k4_fermat'

# GA Parameters
POPULATION_SIZE = 800
GENOTYPE_SHAPE = (3, 25)
NUM_GENES = GENOTYPE_SHAPE[0] * GENOTYPE_SHAPE[1]
NUM_GENERATIONS = 400

TRANSITION_GENERATION = 300
# Exploration Phase Settings
TOURNEY_SIZE_EXPLORE = 3
MUTATION_RATE_EXPLORE = 2.5 / NUM_GENES  # Higher rate
ETA_MUTATION_EXPLORE = 10.0
ETA_CROSSOVER_EXPLORE = 5.0
SPECIATION_THRESHOLD_EXPLORE = 2.5
SPECIES_SHARING_RADIUS_EXPLORE = 2.7

# Exploitation Phase Settings
TOURNEY_SIZE_EXPLOIT = 7
MUTATION_RATE_EXPLOIT = 0.5 / NUM_GENES  # Lower rate
ETA_MUTATION_EXPLOIT = 100.0
ETA_CROSSOVER_EXPLOIT = 30.0
SPECIATION_THRESHOLD_EXPLOIT = 1.5 
SPECIES_SHARING_RADIUS_EXPLOIT = 1.7

# Crossover and Mutation Parameters
CROSSOVER_RATE = 0.9

STAGNATION_THRESHOLD = 20  # Generations a species can go without improvement before being removed.
SPECIES_ELITISM = 1        # Number of best individuals per species to carry over directly.

# Batching for Fitness Evaluation
FITNESS_MINI_BATCH_SIZE = 50
LOG_INTERVAL = 1

# Checkpointing
CHECKPOINT_DIR = 'checkpoints_3k'
CHECKPOINT_INTERVAL = 100

#MINSET_SIZE = 100000
#NEWTON_STEPS = 100
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
        coeffs: (3, 25) coefficient array
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
        return f"Species(id={self.id}, members={len(self.members)}, best_fitness={self.best_fitness:.4f}, stagnated_for={self.generations_since_improvement})"

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
def polynomial_mutation(key, individual, prob_mut, eta):
    key1, key2 = jax.random.split(key)
    u = jax.random.uniform(key1, shape=individual.shape)
    do_mutation = jax.random.uniform(key2, shape=individual.shape) < prob_mut
    delta = jnp.where(u < 0.5, (2 * u)**(1 / (eta + 1)) - 1, 1 - (2 * (1 - u))**(1 / (eta + 1)))
    mutated_individual = jnp.where(do_mutation, individual + delta, individual)
    return jnp.clip(mutated_individual, -1.0, 1.0)

@partial(jit, static_argnames=('max_offspring', 'k_tournament', 'p_mut', 'eta_cross', 'eta_mut'))
def generate_padded_offspring_batch(key, members, fitness, max_offspring, k_tournament, p_mut, eta_cross, eta_mut):
    """Generates a fixed-size batch of offspring and we slice from it later."""
    
    # Create 4 master keys, one for each stochastic operation.
    key_p1, key_p2, key_cross, key_mut = jax.random.split(key, 4)

    # Split each master key into a batch of size max_offspring
    p1_keys = jax.random.split(key_p1, max_offspring)
    p2_keys = jax.random.split(key_p2, max_offspring)
    cross_keys = jax.random.split(key_cross, max_offspring)
    mut_keys = jax.random.split(key_mut, max_offspring)

    # VMAP to perform batched tournament selection
    select_fn = partial(tournament_selection, population=members, fitness=fitness, k=k_tournament)
    parent1_batch = vmap(select_fn)(p1_keys)
    parent2_batch = vmap(select_fn)(p2_keys)

    # VMAP to perform batched crossover
    crossover_fn = partial(sbx_crossover, eta=eta_cross)
    offspring1_batch, offspring2_batch = vmap(crossover_fn)(cross_keys, parent1_batch, parent2_batch)
    
    # Decide which children to keep based on crossover rate
    crossover_mask = vmap(jax.random.uniform)(cross_keys).reshape(-1, 1, 1) < CROSSOVER_RATE
    child_batch = jnp.where(crossover_mask, offspring1_batch, parent1_batch)

    # VMAP to perform batched mutation
    mutation_fn = partial(polynomial_mutation, prob_mut=p_mut, eta=eta_mut)
    mutated_batch = vmap(mutation_fn)(mut_keys, child_batch)

    # VMAP to normalize the final batch
    normalized_batch = vmap(lambda p: normalize_coeffs(canonicalize_coeffs(p)))(mutated_batch)
    
    return normalized_batch


def reproduce_within_species(key, species, num_offspring, tournament_size, eta_mutation, eta_crossover, mutation_rate):
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

    MAX_OFFSPRING_PER_SPECIES = 64 

    if num_members < 2:
        mutation_fn = partial(polynomial_mutation, prob_mut=mutation_rate, eta=eta_mutation)
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
            tournament_size, mutation_rate, eta_crossover, eta_mutation
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
    args = parser.parse_args()
    print("--- Speciation-based GA with Adaptive Schedule ---")
    print(f"Population: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}, Exploration Speciation Threshold: {SPECIATION_THRESHOLD_EXPLORE}, Exploitation Speciation Threshold: {SPECIATION_THRESHOLD_EXPLOIT}")
    print(f"Switching to exploitation mode at generation {TRANSITION_GENERATION}")


    # --- Load points ---
    with open(CYPOINTSFILE, 'rb') as f:
        points_real = np.asarray(pickle.load(f))
    points_real = np.concatenate([np.real(points_real), np.imag(points_real)], axis=1)
    points_real = jax.device_put(jnp.asarray(points_real))

    # --- Create the vmapped fitness function ---
    vmap_fitness_batch = vmap(
        calculate_fitness_for_one_individual,
        in_axes=(0, None, None, None, None, None), out_axes=0
    )

    # --- Load from checkpoint or initialize ---
    start_gen = 0
    population = None
    species_list = []
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
        population = jax.random.uniform(subkey, (POPULATION_SIZE, *GENOTYPE_SHAPE), minval=-1.0, maxval=1.0)
        population = vmap(lambda p: normalize_coeffs(canonicalize_coeffs(p)))(population)
        species_list = []
    
    print(f"\nStarting evolution from generation {start_gen}...")
    start_time = time.time()
    last_log_time = start_time # Initialize timer for logging intervals
    
    # --- Main Evolution Loop ---
    for gen in range(start_gen, NUM_GENERATIONS):
        
        # --- Set parameters based on the current generation ---
        if gen < TRANSITION_GENERATION:
            current_tourney_size = TOURNEY_SIZE_EXPLORE
            current_eta_mutation = ETA_MUTATION_EXPLORE
            current_eta_crossover = ETA_CROSSOVER_EXPLORE
            current_mutation_rate = MUTATION_RATE_EXPLORE
            current_speciation_threshold = SPECIATION_THRESHOLD_EXPLORE
            current_species_sharing_radius = SPECIES_SHARING_RADIUS_EXPLORE
            
        else:
            current_tourney_size = TOURNEY_SIZE_EXPLOIT
            current_eta_mutation = ETA_MUTATION_EXPLOIT
            current_eta_crossover = ETA_CROSSOVER_EXPLOIT
            current_mutation_rate = MUTATION_RATE_EXPLOIT
            current_speciation_threshold = SPECIATION_THRESHOLD_EXPLOIT
            current_species_sharing_radius = SPECIES_SHARING_RADIUS_EXPLOIT
            
        # 1. Calculate fitness for the entire population
        all_fitness_scores = jnp.zeros(POPULATION_SIZE)
        num_batches = (POPULATION_SIZE + FITNESS_MINI_BATCH_SIZE - 1) // FITNESS_MINI_BATCH_SIZE
        for i in range(num_batches):
            start_idx = i * FITNESS_MINI_BATCH_SIZE
            end_idx = min(start_idx + FITNESS_MINI_BATCH_SIZE, POPULATION_SIZE)
            pop_batch = population[start_idx:end_idx]
            fitness_batch = vmap_fitness_batch(
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

        # Create a matrix of all species representatives
        representatives = jnp.array([s.representative for s in species_list])
        
        # Create a vectorized distance function
        dist_to_reps = vmap(calculate_distance, in_axes=(None, 0)) # ind vs all reps
        dist_matrix = vmap(dist_to_reps, in_axes=(0, None))(population, representatives) # all inds vs all reps
        
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

        # 3. Calculate offspring allocation
        # --- NEW: Species-Level Fitness Sharing ---
        
        # Step 1: Calculate the raw average fitness for each species
        raw_avg_fitness = jnp.array([jnp.mean(jnp.array(s.fitness_values)) if s.members else 0.0 for s in species_list])
        
        # Step 2: Calculate distances between all species representatives
        # We already have the 'representatives' array from the speciation step.
        representatives = jnp.array([s.representative for s in species_list])
        dist_to_reps = vmap(calculate_distance, in_axes=(None, 0))
        species_dist_matrix = vmap(dist_to_reps, in_axes=(0, None))(representatives, representatives)

        # Step 3: Calculate niche crowding for each species
        # A species is "crowded" by another if the distance is < the sharing radius.
        # We get a boolean matrix of shape (num_species, num_species).
        sharing_matrix = species_dist_matrix < current_species_sharing_radius
        
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
                        current_tourney_size, current_eta_mutation, current_eta_crossover, current_mutation_rate
                    )
                    next_generation_population.extend(offspring)

        # 4. Handle stagnation and create new population
        for s in species_list: s.update_stagnation()
        
        # Prune stale species, but keep at least one
        species_list = [s for s in species_list if (s.generations_since_improvement < STAGNATION_THRESHOLD or len(species_list) == 1) and s.members]
        
        # Ensure population size is maintained
        if len(next_generation_population) != POPULATION_SIZE:
             # This can happen due to rounding or empty species. Refill if necessary.
             current_pop_size = len(next_generation_population)
             if current_pop_size < POPULATION_SIZE:
                 key, subkey = jax.random.split(key)
                 randoms_needed = POPULATION_SIZE - current_pop_size
                 random_individuals = jax.random.uniform(subkey, (randoms_needed, *GENOTYPE_SHAPE), minval=-1.0, maxval=1.0)
                 random_individuals = vmap(lambda p: normalize_coeffs(canonicalize_coeffs(p)))(random_individuals)
                 next_generation_population.extend(random_individuals)

        population = jnp.array(next_generation_population[:POPULATION_SIZE])
        
        # 5. Logging
        if (gen + 1) % LOG_INTERVAL == 0:
            current_time = time.time()
            duration_for_interval = current_time - last_log_time
            avg_time_per_gen = duration_for_interval / LOG_INTERVAL

            max_fitness = jnp.max(all_fitness_scores)
            avg_fitness = jnp.mean(all_fitness_scores)
            print(f"Gen {gen+1:4d}/{NUM_GENERATIONS} | Species: {len(species_list):2d} | Max Fit: {max_fitness:.4f} | Avg Fit: {avg_fitness:.4f} | Avg Gen Time: {avg_time_per_gen:.2f}s")
        
            last_log_time = current_time # Reset timer for the next interval
        # 6. Checkpointing
        if (gen + 1) % CHECKPOINT_INTERVAL == 0 and (gen + 1) < NUM_GENERATIONS:
            checkpoint_filename = os.path.join(CHECKPOINT_DIR, f'checkpoint_gen_{gen+1}.pkl')
            # Prune members from species before saving to reduce file size
            # The representative and stagnation state is the important part
            species_to_save = [Species(s.representative) for s in species_list]
            for i, s in enumerate(species_to_save):
                s.id = species_list[i].id
                s.best_fitness = species_list[i].best_fitness
                s.generations_since_improvement = species_list[i].generations_since_improvement

            checkpoint_data = {
                'population': population,
                'generation': gen,
                'key': key,
                'species_list': species_to_save
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
        end_idx = jnp.minimum(start_idx + FITNESS_MINI_BATCH_SIZE, POPULATION_SIZE)
        population_batch = population[start_idx:end_idx]
        fitness_batch = vmap_fitness_batch(population_batch, points_real, PSI, MINSET_SIZE, NEWTON_STEPS, METRIC)
        final_fitness = final_fitness.at[start_idx:end_idx].set(fitness_batch)

    # 2. Use the correct vectorized speciation to assign members.
    for s in species_list: s.clear_members()
    
    if species_list:
        representatives = jnp.array([s.representative for s in species_list])
        
        dist_to_reps = vmap(calculate_distance, in_axes=(None, 0))
        dist_matrix = vmap(dist_to_reps, in_axes=(0, None))(population, representatives)
        
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

    rank = 1
    for s in final_species_list:
        best_member_idx = jnp.argmax(jnp.array(s.fitness_values))
        best_member = s.members[best_member_idx]
        best_fitness = s.fitness_values[best_member_idx]

        print(f"\n--- Species {s.id} (Best Fitness: {best_fitness:.5f}) ---")
        print(f"Size: {len(s.members)} members | Stagnated for: {s.generations_since_improvement} gens")
        print("Best Member's Coefficients:")
        print(format_array_with_commas(best_member))
        print("Complex equations:")
        print(combine_to_complex_equations(get_basis_labels(), best_member))

        parent_folder = os.path.join(
            f'plots_slag_{args.job_id}', 
            f'plots_slag_{args.job_id}_{rank}_id{s.id}'
        )

        make_fitness_plots(points_real, best_member, PSI, k=100000, n_refine_steps=100, metric=METRIC, compare_with_random=False, parent_folder=parent_folder)
        rank += 1
