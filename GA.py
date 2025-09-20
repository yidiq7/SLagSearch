import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from functools import partial
import pickle
import time
import os
import argparse
from find_smooth_submanifold import filter_and_refine, normalize_coeffs, get_basis_labels, combine_to_complex_equations
from slag_condition import compute_combined_fitness
from helper import canonicalize_coeffs, format_array_with_commas

jax.config.update('jax_default_matmul_precision', 'highest')

# -----------------------------------------------------------------------------
# 1. HYPERPARAMETERS
# -----------------------------------------------------------------------------
# Moduli of the quintic
#PSI = 1000000
#CYPOINTSFILE = f'/projects/ruehlehet/yidi/sLag/data_psi/5mil_patch0_psi{PSI}_seed1024.pkl'
PSI = 0
CYPOINTSFILE = '/projects/ruehlehet/yidi/sLag/data/5mil_patch0_343.pkl'

# GA Parameters
POPULATION_SIZE = 400      # Size of the population.
GENOTYPE_SHAPE = (3, 25)   # Shape of a single individual's genotype.
NUM_GENES = GENOTYPE_SHAPE[0] * GENOTYPE_SHAPE[1]
NUM_GENERATIONS = 400   # Total number of generations to run.
TOURNAMENT_SIZE = 3       # Number of individuals selected for a tournament.

# Crossover and Mutation Parameters
CROSSOVER_RATE = 0.9       # Probability of performing crossover.
MUTATION_RATE = 1.5 / NUM_GENES # Probability of mutating a single gene.
ETA_CROSSOVER = 15.0       # Distribution index for Simulated Binary Crossover (SBX).
ETA_MUTATION = 20.0        # Distribution index for Polynomial Mutation.

# Niching Parameters
SIGMA_SHARE = 0.25         # The radius of a niche.
ALPHA_SHARE = 1.0          # Shape parameter for the sharing function.

# Batching for Fitness Evaluation
# This is the key parameter to control memory usage.
# It's the largest number of individuals you can evaluate at once without an OOM error.
# Tune this based on your GPU's VRAM (e.g., 4, 8, 16, 32).
FITNESS_MINI_BATCH_SIZE = 50
LOG_INTERVAL = 1         # How often to print progress (in generations).

CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_INTERVAL = 100  # Save progress every 100 generations

MINSET_SIZE = 10000
NEWTON_STEPS = 40

# JAX PRNG Key
key = jax.random.PRNGKey(42)


# -----------------------------------------------------------------------------
# 2. CORE EVALUATION FUNCTIONS (FITNESS, NORMALIZATION, REFINEMENT)
# -----------------------------------------------------------------------------
# --- Combined Fitness Evaluation for a Single Individual ---

@partial(jit, static_argnames=('k', 'n_refine_steps', 'constant_coord'))
def calculate_fitness_for_one_individual(coeffs: jnp.ndarray, points_real: jnp.ndarray, psi: jnp.ndarray, k: int, n_refine_steps: int, constant_coord: int = 0) -> jnp.float32:
    """
    This function chains the refinement and fitness calculation for one individual.
    It is this entire unit that will be batched using vmap.
    """
    min_set_real = filter_and_refine(points_real, coeffs, psi, k, n_refine_steps, constant_coord)
    fitness = compute_combined_fitness(min_set_real, coeffs, psi, constant_coord)
    return fitness


# -----------------------------------------------------------------------------
# 3. NICHING IMPLEMENTATION (FITNESS SHARING)
# -----------------------------------------------------------------------------

@jit
def calculate_distance(ind1, ind2):
    """Calculates Euclidean distance between two flattened individuals."""
    return jnp.linalg.norm(ind1.ravel() - ind2.ravel())

@partial(jit, static_argnames=('sigma', 'alpha'))
def sharing_function(distance, sigma, alpha):
    """Calculates the sharing value based on distance."""
    return jnp.maximum(0, 1 - (distance / sigma)**alpha)

@partial(jit, static_argnames=('sigma', 'alpha'))
def get_shared_fitness(population, raw_fitness, sigma, alpha):
    """Adjusts raw fitness scores based on niche counts."""
    dist_matrix = vmap(lambda ind1: vmap(lambda ind2: calculate_distance(ind1, ind2))(population))(population)
    sharing_matrix = sharing_function(dist_matrix, sigma, alpha)
    niche_counts = jnp.sum(sharing_matrix, axis=1)
    niche_counts = jnp.maximum(niche_counts, 1.0)
    shared_fitness = raw_fitness / niche_counts
    return shared_fitness


# -----------------------------------------------------------------------------
# 4. GENETIC ALGORITHM OPERATORS
# -----------------------------------------------------------------------------

@partial(jit, static_argnames=('k',))
def tournament_selection(key, population, fitness, k):
    """Selects an individual using tournament selection."""
    indices = jax.random.randint(key, (k,), 0, POPULATION_SIZE)
    participants_fitness = fitness[indices]
    winner_index_in_tournament = jnp.argmax(participants_fitness)
    return population[indices[winner_index_in_tournament]]

@partial(jit, static_argnames=('eta',))
def sbx_crossover(key, parent1, parent2, eta):
    """Simulated Binary Crossover (SBX) for two individuals."""
    u = jax.random.uniform(key, shape=parent1.shape)
    beta = jnp.where(u <= 0.5, (2 * u)**(1 / (eta + 1)), (1 / (2 * (1 - u)))**(1 / (eta + 1)))
    offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    offspring1 = jnp.clip(offspring1, -1.0, 1.0)
    offspring2 = jnp.clip(offspring2, -1.0, 1.0)
    return offspring1, offspring2

@partial(jit, static_argnames=('prob_mut', 'eta'))
def polynomial_mutation(key, individual, prob_mut, eta):
    """Applies polynomial mutation to an individual."""
    key1, key2 = jax.random.split(key)
    u = jax.random.uniform(key1, shape=individual.shape)
    do_mutation = jax.random.uniform(key2, shape=individual.shape) < prob_mut
    delta = jnp.where(u < 0.5, (2 * u)**(1 / (eta + 1)) - 1, 1 - (2 * (1 - u))**(1 / (eta + 1)))
    mutated_individual = jnp.where(do_mutation, individual + delta, individual)
    return jnp.clip(mutated_individual, -1.0, 1.0)


# -----------------------------------------------------------------------------
# 5. MAIN GA LOOP
# -----------------------------------------------------------------------------

@jax.jit
def ga_step(key, population, fitness):
    """Performs one full generation of the Genetic Algorithm."""
    next_pop = jnp.empty_like(population)
    shared_fitness = get_shared_fitness(population, fitness, SIGMA_SHARE, ALPHA_SHARE)
    keys = jax.random.split(key, POPULATION_SIZE // 2 + 1)

    def evolution_loop_body(i, state):
        current_pop, loop_keys = state
        selection_key1, selection_key2, crossover_key, mutation_key1, mutation_key2 = jax.random.split(loop_keys[i], 5)
        
        parent1 = tournament_selection(selection_key1, population, shared_fitness, TOURNAMENT_SIZE)
        parent2 = tournament_selection(selection_key2, population, shared_fitness, TOURNAMENT_SIZE)
        
        do_crossover = jax.random.uniform(crossover_key) < CROSSOVER_RATE
        offspring1, offspring2 = sbx_crossover(crossover_key, parent1, parent2, ETA_CROSSOVER)
        child1 = jax.lax.cond(do_crossover, lambda: offspring1, lambda: parent1)
        child2 = jax.lax.cond(do_crossover, lambda: offspring2, lambda: parent2)
        
        mutated_child1 = polynomial_mutation(mutation_key1, child1, MUTATION_RATE, ETA_MUTATION)
        mutated_child2 = polynomial_mutation(mutation_key2, child2, MUTATION_RATE, ETA_MUTATION)
        
        normalized_child1 = normalize_coeffs(canonicalize_coeffs(mutated_child1))
        normalized_child2 = normalize_coeffs(canonicalize_coeffs(mutated_child2))
        
        new_pop = current_pop.at[2*i].set(normalized_child1)
        new_pop = new_pop.at[2*i+1].set(normalized_child2)
        return (new_pop, loop_keys)

    final_pop, _ = jax.lax.fori_loop(0, POPULATION_SIZE // 2, evolution_loop_body, (next_pop, keys))
    
    if POPULATION_SIZE% 2 == 1:
        last_key = keys[-1]
        selection_key, mutation_key = jax.random.split(last_key)
        parent = tournament_selection(selection_key, population, shared_fitness, TOURNAMENT_SIZE)
        mutated_parent = polynomial_mutation(mutation_key, parent, MUTATION_RATE, ETA_MUTATION)
        normalized_parent = normalize_coeffs(canonicalize_coeffs(mutated_parent))
        final_pop = final_pop.at[-1].set(normalized_parent)

    return final_pop


# -----------------------------------------------------------------------------
# 6. EXECUTION
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # ---  Argument Parsing for Checkpoint Control ---
    parser = argparse.ArgumentParser(description="Run a Genetic Algorithm with advanced checkpointing.")
    parser.add_argument(
        '--load_checkpoint',
        type=str,
        nargs='?',
        const='latest',
        default=None,
        help="Load a checkpoint. Use 'latest' to load the most recent, or provide a specific filename. No argument starts a fresh run."
    )
    args = parser.parse_args()

    print("--- GA with Niching and Batched Fitness Evaluation in JAX ---")
    print(f"Population Size: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}")
    print(f"Fitness Mini-Batch Size: {FITNESS_MINI_BATCH_SIZE}")

    # --- Load points ---
    with open(CYPOINTSFILE, 'rb') as f:
        pts_5mil_patch0 = pickle.load(f)

    pts_5mil_patch0 = np.asarray(pts_5mil_patch0)
    points_real = np.concatenate([np.real(pts_5mil_patch0), np.imag(pts_5mil_patch0)], axis=1)
    points_real = jnp.asarray(points_real)


    # --- Create the vmapped function for batch evaluation ---
    vmap_fitness_batch = vmap(
        calculate_fitness_for_one_individual,
        in_axes=(0, None, None, None, None, None), 
        out_axes=0
    )
    
   # ---  Advanced Load from checkpoint or initialize ---
    start_gen = 0
    population = None
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint_to_load = None
    if args.load_checkpoint == 'latest':
        # Find the checkpoint with the highest generation number
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
        # Load a specific file
        checkpoint_to_load = os.path.join(CHECKPOINT_DIR, args.load_checkpoint)

    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        print(f"\nCheckpoint file found. Resuming run from: {checkpoint_to_load}")
        with open(checkpoint_to_load, 'rb') as f:
            checkpoint = pickle.load(f)
        population = checkpoint['population']
        start_gen = checkpoint['generation'] + 1
        key = checkpoint['key']
        population = jnp.asarray(population) # Ensure it's a JAX array
    else:
        if args.load_checkpoint:
            print(f"Warning: Checkpoint '{args.load_checkpoint}' not found.")
        print("\nNo valid checkpoint specified. Starting a new run.")
        key, subkey = jax.random.split(key)
        population = jax.random.uniform(subkey, (POPULATION_SIZE, *GENOTYPE_SHAPE), minval=-1.0, maxval=1.0)
        print("Normalizing initial population...")
        population = vmap(lambda p: normalize_coeffs(canonicalize_coeffs(p)))(population)
   
    print(f"\nStarting evolution from generation {start_gen}...")
    start_time = time.time()
    last_log_time = start_time # Initialize timer for logging intervals

    for gen in range(start_gen, NUM_GENERATIONS):
        key, subkey = jax.random.split(key)

        # --- Batched Fitness Evaluation ---
        all_fitness_scores = jnp.zeros(POPULATION_SIZE)
        num_batches = (POPULATION_SIZE + FITNESS_MINI_BATCH_SIZE - 1) // FITNESS_MINI_BATCH_SIZE

        for i in range(num_batches):
            start_idx = i * FITNESS_MINI_BATCH_SIZE
            end_idx = jnp.minimum(start_idx + FITNESS_MINI_BATCH_SIZE, POPULATION_SIZE)
            population_batch = population[start_idx:end_idx]
            fitness_batch = vmap_fitness_batch(
                population_batch, points_real, PSI,
                MINSET_SIZE, NEWTON_STEPS, 0
            )
            all_fitness_scores = all_fitness_scores.at[start_idx:end_idx].set(fitness_batch)

        # --- Evolve to Next Generation ---
        population = ga_step(subkey, population, all_fitness_scores)
        # --- Logging and Timing ---
        if (gen + 1) % LOG_INTERVAL == 0:
            current_time = time.time()
            duration_for_interval = current_time - last_log_time
            avg_time_per_gen = duration_for_interval / LOG_INTERVAL
            
            max_fitness = jnp.max(all_fitness_scores)
            avg_fitness = jnp.mean(all_fitness_scores)
            
            print(f"Generation {gen+1:4d}/{NUM_GENERATIONS} | Max Fitness: {max_fitness:.4f} | Avg Fitness: {avg_fitness:.4f} | Avg Gen Time: {avg_time_per_gen:.2f}s")
            
            last_log_time = current_time # Reset timer for the next interval

        # --- Save Named Checkpoint ---
        if (gen + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_filename = os.path.join(CHECKPOINT_DIR, f'checkpoint_gen_{gen+1}.pkl')
            checkpoint_data = {
                'population': population,
                'generation': gen,
                'key': key
            }
            with open(checkpoint_filename, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"--- Checkpoint saved at generation {gen+1} to {checkpoint_filename} ---")


    end_time = time.time()
    print(f"\nEvolution finished in {end_time - start_time:.2f} seconds.")

    # --- Final Analysis ---
    print("\n--- Analyzing Final Population ---")
    # Re-calculate final fitness for analysis
    final_fitness = jnp.zeros(POPULATION_SIZE)
    num_batches = (POPULATION_SIZE + FITNESS_MINI_BATCH_SIZE - 1) // FITNESS_MINI_BATCH_SIZE
    for i in range(num_batches):
        start_idx = i * FITNESS_MINI_BATCH_SIZE
        end_idx = jnp.minimum(start_idx + FITNESS_MINI_BATCH_SIZE, POPULATION_SIZE)
        population_batch = population[start_idx:end_idx]
        fitness_batch = vmap_fitness_batch(population_batch, points_real, PSI, MINSET_SIZE, NEWTON_STEPS, 0)
        final_fitness = final_fitness.at[start_idx:end_idx].set(fitness_batch)

    sorted_indices = jnp.argsort(final_fitness)[::-1]
    sorted_population = population[sorted_indices]
    sorted_fitness = final_fitness[sorted_indices]
    
    print("\nIdentifying distinct solutions (niches)...")
    labels = get_basis_labels()
    niche_representatives = []
    
    for i in range(POPULATION_SIZE):
        individual = sorted_population[i]
        fitness = sorted_fitness[i]
        is_in_existing_niche = False
        
        for rep in niche_representatives:
            if calculate_distance(individual, rep) < SIGMA_SHARE:
                is_in_existing_niche = True
                break
        
        if not is_in_existing_niche:
            niche_representatives.append(individual)
            print(f"\nFound new niche representative with fitness: {fitness:.5f}")
            print("Coefficients for this niche:")
            print(individual)
            print("Array form ready to copy:")
            print(format_array_with_commas(individual))
            print("Complex equations:")
            print(combine_to_complex_equations(labels, individual)) 

    print(f"\nFound {len(niche_representatives)} distinct niches in the final population.")

