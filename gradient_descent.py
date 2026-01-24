import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
import argparse
import os
from functools import partial
import optax # Restore Optax

from find_smooth_submanifold import compute_distances_batched, normalize_coeffs, evaluate_equations_single_point, compute_affine_jacobian, convert_real_to_complex_single, determine_patch_and_rescale_single, convert_complex_to_real_single, PATCH_ACTIVE_INDICES
from slag_condition import compute_combined_fitness, compute_special_condition_fitness_smooth
from helper import canonicalize_coeffs

# Enable 64-bit precision (Critical for stability)
jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
PSI = 0
CYPOINTSFILE = f'/projects/ruehlehet/yidi/sLag/data_psi/1mil_patch_all_psi{PSI}_seed1024.pkl'
METRIC = 'k4_fermat'

# Optimization Parameters
LEARNING_RATE = 0.001 # Adam handles scale, so standard LR is fine
NUM_STEPS = 40
MINSET_SIZE = 10000
NEWTON_STEPS = 100
MINE_INTERVAL = 20 

# -----------------------------------------------------------------------------
# SIMPLIFIED NEWTON SOLVER
# -----------------------------------------------------------------------------
def refine_point_iterative_simple(
    p_10d_init: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    n_steps: int
) -> jnp.ndarray:
    
    alpha = 1.0 
    p_complex_init = convert_real_to_complex_single(p_10d_init)
    _, patch_index_init = determine_patch_and_rescale_single(p_complex_init)
    init_state = (p_10d_init, patch_index_init)

    def body_fn(i, state):
        p_10d, patch_index = state
        active_indices = PATCH_ACTIVE_INDICES[patch_index]

        f_vec = evaluate_equations_single_point(p_10d, coeffs, psi)
        J = compute_affine_jacobian(p_10d, patch_index, coeffs, psi)
        
        # Lower regularization now that we have x64
        JJT = J @ J.T + 1e-8 * jnp.eye(J.shape[0])
        
        w = jnp.linalg.solve(JJT, -f_vec)
        delta_p_active = J.T @ w

        p_10d = p_10d.at[active_indices].add(alpha * delta_p_active)

        p_complex = convert_real_to_complex_single(p_10d)
        p_complex_rescaled, patch_index = determine_patch_and_rescale_single(p_complex)
        p_10d_rescaled = convert_complex_to_real_single(p_complex_rescaled) 

        return (p_10d_rescaled, patch_index)
    
    p_10d_final, _ = jax.lax.fori_loop(0, n_steps, body_fn, init_state)
    return p_10d_final

# -----------------------------------------------------------------------------
# 2. MINING STEP
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=('k',))
def mine_indices(
    coeffs: jnp.ndarray,
    points_real: jnp.ndarray,
    psi: jnp.ndarray,
    k: int
) -> jnp.ndarray:
    all_distances = compute_distances_batched(points_real, coeffs, psi)
    best_indices = jnp.argsort(all_distances)[:k]
    return best_indices

# -----------------------------------------------------------------------------
# 3. LOSS FUNCTION
# -----------------------------------------------------------------------------
def compute_loss_on_fixed_points(
    coeffs: jnp.ndarray,
    fixed_points_real: jnp.ndarray,
    psi: jnp.ndarray,
    n_refine_steps: int,
    metric: str
) -> tuple[jnp.float32, tuple[jnp.float32, jnp.float32]]:
    
    # Use the SIMPLIFIED solver
    refine_fn = partial(refine_point_iterative_simple, coeffs=coeffs, psi=psi, n_steps=n_refine_steps)
    min_set_real = jax.vmap(refine_fn)(fixed_points_real)

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

    # Lagrangian Loss
    kahler_form_restricted = jnp.einsum('nij,nik,njl->nkl', kahler_form_unrestricted, restrictions, restrictions)
    frobenius_norms = jnp.linalg.norm(kahler_form_restricted, axis=(1, 2))
    normalization_factor = jnp.linalg.norm(kahler_form_unrestricted, axis=(1, 2))
    
    norms_normalized = frobenius_norms / (normalization_factor + 1e-9)
    
    # Use sorting again to be robust to outliers
    sorted_norms = jnp.sort(norms_normalized)
    cutoff_index = int(sorted_norms.shape[0] * 0.99)
    norms_cut = sorted_norms[:cutoff_index]
    
    lagrangian_loss = jnp.mean(norms_cut)

    # Special Loss
    phases = compute_holomorphic_form_restricted(
        min_set, patch_indices, restrictions, phase_only=True
    )
    order_parameter = compute_special_condition_fitness_smooth(phases)
    special_loss = 1.0 - order_parameter
    
    # Combine both
    total_loss = lagrangian_loss + special_loss
    
    return total_loss, (lagrangian_loss, special_loss)

loss_value_and_grad = jax.jit(jax.value_and_grad(compute_loss_on_fixed_points, argnums=0, has_aux=True), static_argnames=('n_refine_steps', 'metric'))

# -----------------------------------------------------------------------------
# 4. MAIN LOOP (Adam)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gradient Descent for sLag Search")
    parser.add_argument("--psi", type=int, default=0, help="PSI parameter")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--steps", type=int, default=1000, help="Number of optimization steps")
    parser.add_argument("--job_id", type=str, default="0", help="Job ID for saving results")
    args = parser.parse_args()

    PSI = args.psi
    LEARNING_RATE = args.lr
    NUM_STEPS = args.steps
    
    print(f"--- Hybrid Mining-Adam Optimization (Stable) ---")
    print(f"PSI: {PSI}, LR: {LEARNING_RATE}, Steps: {NUM_STEPS}")
    
    # Update global PSI if needed, or pass it down. 
    # Since PSI is used in mine_indices and loss, we should pass it.
    # The global constant PSI was used in main() before. We update it here for the main scope.
    
    # Load Data
    # Note: Using the global variable name for file path which depends on PSI
    cypoints_file = f'/projects/ruehlehet/yidi/sLag/data_psi/1mil_patch_all_psi{PSI}_seed1024.pkl'
    
    try:
        with open(cypoints_file, 'rb') as f:
            points_real = np.asarray(pickle.load(f))
        points_real = np.concatenate([np.real(points_real), np.imag(points_real)], axis=1)
        points_real = jax.device_put(jnp.asarray(points_real))
        print(f"Loaded {len(points_real)} points from {cypoints_file}.")
    except FileNotFoundError:
        print(f"Warning: Data file {cypoints_file} not found. Using random points.")
        key = jax.random.PRNGKey(0)
        random_complex = jax.random.normal(key, (100000, 5), dtype=jnp.complex64)
        random_complex = random_complex / jnp.linalg.norm(random_complex, axis=1, keepdims=True)
        points_real = jnp.concatenate([jnp.real(random_complex), jnp.imag(random_complex)], axis=1)

    # Init Coeffs (New Seed)
    key = jax.random.PRNGKey(int(args.job_id) if args.job_id.isdigit() else 42) 
    coeffs = jax.random.uniform(key, (3, 25), minval=-1.0, maxval=1.0)
    coeffs = normalize_coeffs(coeffs)
    
    # Init Optimizer
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(coeffs)

    print(f"Starting {NUM_STEPS} steps (Re-mining every {MINE_INTERVAL} steps)...")
    
    current_batch = None
    
    for step in range(NUM_STEPS):
        start_time = time.time()
        
        # Mining
        if step % MINE_INTERVAL == 0:
            mine_start = time.time()
            active_indices = mine_indices(coeffs, points_real, PSI, MINSET_SIZE)
            current_batch = points_real[active_indices]
            mine_time = time.time() - mine_start
            print(f"  [Mining] Selected new {MINSET_SIZE} points in {mine_time:.2f}s")
            
        # Training
        (loss_val, (lag_loss, spec_loss)), grads = loss_value_and_grad(
            coeffs, current_batch, PSI, NEWTON_STEPS, METRIC
        )
        
        # We don't need manual clipping with Adam usually, but let's be safe against infs
        # Replace infs/nans with zero in gradients
        grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        
        # --- Tangent Projection (Crucial for Spherical Optimization) ---
        # Ensure gradient effectively only moves coeffs along the sphere surface
        dot_prods = jnp.sum(grads * coeffs, axis=1, keepdims=True)
        grads = grads - dot_prods * coeffs
        
        updates, opt_state = optimizer.update(grads, opt_state, coeffs)
        coeffs = optax.apply_updates(coeffs, updates)
        
        # Renormalize to stay on sphere
        coeffs = normalize_coeffs(coeffs)
        
        epoch_time = time.time() - start_time
        grad_norm = jnp.linalg.norm(grads)
        print(f"Step {step+1:4d} | Total: {loss_val:.6f} | Lag: {lag_loss:.6f} | Spec: {spec_loss:.6f} | GNorm: {grad_norm:.2e} | Time: {epoch_time:.2f}s")

    print("\nOptimization Complete.")
    print("Final Coefficients:")
    print(coeffs)

if __name__ == "__main__":
    main()
