import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
import argparse
import os
from functools import partial
import optax

from find_smooth_submanifold import refine_point_iterative, compute_distances_batched, normalize_coeffs
from slag_condition import compute_combined_fitness, compute_special_condition_fitness_smooth
from helper import canonicalize_coeffs

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
PSI = 0
CYPOINTSFILE = f'/projects/ruehlehet/yidi/sLag/data_psi/1mil_patch_all_psi{PSI}_seed1024.pkl'
METRIC = 'k4_fermat'

LEARNING_RATE = 0.001
NUM_STEPS = 1000
MINSET_SIZE = 10000
NEWTON_STEPS = 40

# Re-mining frequency: How often to pick a new set of points
MINE_INTERVAL = 20 

# -----------------------------------------------------------------------------
# 2. MINING STEP (Non-Differentiable Selection)
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=('k',))
def mine_indices(
    coeffs: jnp.ndarray,
    points_real: jnp.ndarray,
    psi: jnp.ndarray,
    k: int
) -> jnp.ndarray:
    """
    Finds the indices of the k points closest to the manifold defined by coeffs.
    This step is NOT differentiable and is run periodically.
    """
    # Compute initial distances to the current manifold guess
    all_distances = compute_distances_batched(points_real, coeffs, psi)
    
    # Select best k points
    # We select 2*k and refine them, then pick best k?
    # Or just pick best k initial candidates?
    # Let's mirror the original logic: Pick best 2*k, refine once, then pick best k.
    # But for speed in the loop, let's just pick the best K initial candidates for now.
    # If the points are already somewhat close, this is fine.
    best_indices = jnp.argsort(all_distances)[:k]
    
    return best_indices

# -----------------------------------------------------------------------------
# 3. LOSS FUNCTION (Differentiable on FIXED indices)
# -----------------------------------------------------------------------------
def compute_loss_on_fixed_points(
    coeffs: jnp.ndarray,
    fixed_points_real: jnp.ndarray, # These are the points selected by the miner
    psi: jnp.ndarray,
    n_refine_steps: int,
    metric: str
) -> tuple[jnp.float32, tuple[jnp.float32, jnp.float32]]:
    
    # 1. Refine the FIXED set of points
    # This refinement IS differentiable.
    refine_fn = partial(
        refine_point_iterative,
        coeffs=coeffs,
        psi=psi,
        n_steps=n_refine_steps
    )
    min_set_real = jax.vmap(refine_fn)(fixed_points_real)

    # 2. Compute Fitness
    from helper import convert_real_to_complex_batch, determine_patches_batch
    from slag_condition import (
        vmap_compute_affine_jacobian, 
        vmap_compute_restriction,
        compute_kahler_form_unrestricted, 
        compute_holomorphic_form_restricted
    )

    min_set = convert_real_to_complex_batch(min_set_real)
    patch_indices = determine_patches_batch(min_set) 

    jacobians = vmap_compute_affine_jacobian(min_set_real, patch_indices, coeffs, psi)
    restrictions = vmap_compute_restriction(jacobians)

    kahler_form_unrestricted = compute_kahler_form_unrestricted(min_set, patch_indices, metric=metric)

    # --- Lagrangian Loss ---
    kahler_form_restricted = jnp.einsum('nij,nik,njl->nkl', kahler_form_unrestricted, restrictions, restrictions)
    frobenius_norms = jnp.linalg.norm(kahler_form_restricted, axis=(1, 2))
    normalization_factor = jnp.linalg.norm(kahler_form_unrestricted, axis=(1, 2))
    
    norms_normalized = frobenius_norms / (normalization_factor + 1e-9)
    
    sorted_norms = jnp.sort(norms_normalized)
    cutoff_index = int(sorted_norms.shape[0] * 0.99)
    norms_cut = sorted_norms[:cutoff_index]
    
    lagrangian_loss = jnp.mean(norms_cut)

    # --- Special Loss ---
    phases = compute_holomorphic_form_restricted(
        min_set, patch_indices, restrictions, phase_only=True
    )
    order_parameter = compute_special_condition_fitness_smooth(phases)
    special_loss = 1.0 - order_parameter
    
    total_loss = lagrangian_loss + special_loss
    
    return total_loss, (lagrangian_loss, special_loss)

loss_value_and_grad = jax.jit(jax.value_and_grad(compute_loss_on_fixed_points, argnums=0, has_aux=True), static_argnames=('n_refine_steps', 'metric'))


# -----------------------------------------------------------------------------
# 4. MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    print("--- Hybrid Mining-Adam Optimization ---")
    
    # Load Data
    try:
        with open(CYPOINTSFILE, 'rb') as f:
            points_real = np.asarray(pickle.load(f))
        points_real = np.concatenate([np.real(points_real), np.imag(points_real)], axis=1)
        points_real = jax.device_put(jnp.asarray(points_real))
        print(f"Loaded {len(points_real)} points.")
    except FileNotFoundError:
        print("Using random points.")
        key = jax.random.PRNGKey(0)
        random_complex = jax.random.normal(key, (100000, 5), dtype=jnp.complex64)
        random_complex = random_complex / jnp.linalg.norm(random_complex, axis=1, keepdims=True)
        points_real = jnp.concatenate([jnp.real(random_complex), jnp.imag(random_complex)], axis=1)

    # Init Coeffs
    key = jax.random.PRNGKey(42)
    coeffs = jax.random.uniform(key, (3, 25), minval=-1.0, maxval=1.0)
    coeffs = normalize_coeffs(coeffs) # Removed RREF as requested
    
    # Init Optimizer
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(coeffs)

    print(f"Starting {NUM_STEPS} steps (Re-mining every {MINE_INTERVAL} steps)...")
    
    current_batch = None
    
    for step in range(NUM_STEPS):
        start_time = time.time()
        
        # --- PHASE 1: MINING (Every N steps) ---
        if step % MINE_INTERVAL == 0:
            mine_start = time.time()
            # Find the best indices for the CURRENT coefficients
            active_indices = mine_indices(coeffs, points_real, PSI, MINSET_SIZE)
            # Freeze the actual coordinate values of these points to be used for the next N steps
            current_batch = points_real[active_indices]
            mine_time = time.time() - mine_start
            print(f"  [Mining] Selected new {MINSET_SIZE} points in {mine_time:.2f}s")
            
        # --- PHASE 2: TRAINING (Every step) ---
        # Optimize coeffs to better fit the CURRENT BATCH
        (loss_val, (lag_loss, spec_loss)), grads = loss_value_and_grad(
            coeffs, current_batch, PSI, NEWTON_STEPS, METRIC
        )
        
        updates, opt_state = optimizer.update(grads, opt_state, coeffs)
        coeffs = optax.apply_updates(coeffs, updates)
        coeffs = normalize_coeffs(coeffs)
        
        epoch_time = time.time() - start_time
        
        print(f"Step {step+1:4d} | Total: {loss_val:.6f} | Lag: {lag_loss:.6f} | Spec: {spec_loss:.6f} | Time: {epoch_time:.2f}s")

    print("\nOptimization Complete.")
    print("Final Coefficients:")
    print(coeffs)

if __name__ == "__main__":
    main()
