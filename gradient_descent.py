import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
import argparse
import os
from functools import partial

from find_smooth_submanifold import compute_distances_batched, normalize_coeffs, evaluate_equations_single_point, compute_affine_jacobian, convert_real_to_complex_single, determine_patch_and_rescale_single, convert_complex_to_real_single, PATCH_ACTIVE_INDICES
from slag_condition import compute_combined_fitness, compute_special_condition_fitness_smooth
from helper import canonicalize_coeffs

# ... (Configuration remains the same) ...

# -----------------------------------------------------------------------------
# SIMPLIFIED NEWTON SOLVER (AD-Friendly)
# -----------------------------------------------------------------------------
def refine_point_iterative_simple(
    p_10d_init: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    n_steps: int
) -> jnp.ndarray:
    """
    Simplified Newton solver without backtracking line search.
    Fixed step size alpha=1.0 (or smaller) to ensure stable gradients.
    """
    # Use fixed alpha. 1.0 is standard Newton. 0.5 is safer.
    alpha = 1.0 
    
    p_complex_init = convert_real_to_complex_single(p_10d_init)
    _, patch_index_init = determine_patch_and_rescale_single(p_complex_init)
    init_state = (p_10d_init, patch_index_init)

    def body_fn(i, state):
        p_10d, patch_index = state
        active_indices = PATCH_ACTIVE_INDICES[patch_index]

        # Compute Newton step
        f_vec = evaluate_equations_single_point(p_10d, coeffs, psi)
        J = compute_affine_jacobian(p_10d, patch_index, coeffs, psi)
        
        # Increased regularization for stability
        JJT = J @ J.T + 1e-5 * jnp.eye(J.shape[0])
        
        w = jnp.linalg.solve(JJT, -f_vec)
        delta_p_active = J.T @ w

        # Simple update (No backtracking)
        p_10d = p_10d.at[active_indices].add(alpha * delta_p_active)

        # Rescale
        p_complex = convert_real_to_complex_single(p_10d)
        p_complex_rescaled, patch_index = determine_patch_and_rescale_single(p_complex)
        p_10d_rescaled = convert_complex_to_real_single(p_complex_rescaled) 

        return (p_10d_rescaled, patch_index)
    
    p_10d_final, _ = jax.lax.fori_loop(0, n_steps, body_fn, init_state)
    return p_10d_final

# -----------------------------------------------------------------------------
# 2. MINING STEP
# -----------------------------------------------------------------------------
# ... (Mine indices remains the same) ...

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
    
    # --- DUMMY LOSS TEST ---
    # Testing if Newton solver gradients are stable.
    # restrictions = vmap_compute_restriction(jacobians)
    # kahler_form_unrestricted = compute_kahler_form_unrestricted(min_set, patch_indices, metric=metric)
    
    # Simple dummy loss: push points to have smaller norm (meaningless on CP4 but differentiable)
    dummy_loss = jnp.mean(jnp.abs(min_set))
    
    total_loss = dummy_loss
    
    # Return dummy values for aux
    return total_loss, (dummy_loss, 0.0)

loss_value_and_grad = jax.jit(jax.value_and_grad(compute_loss_on_fixed_points, argnums=0, has_aux=True), static_argnames=('n_refine_steps', 'metric'))

# -----------------------------------------------------------------------------
# 4. MAIN LOOP (Manual SGD + Tangent Projection)
# -----------------------------------------------------------------------------
def main():
    print("--- Manual SGD with Tangent Projection ---")
    
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

    # Init Coeffs (New Seed)
    key = jax.random.PRNGKey(123) 
    coeffs = jax.random.uniform(key, (3, 25), minval=-1.0, maxval=1.0)
    coeffs = normalize_coeffs(coeffs)
    
    # Init Momentum Buffer
    velocity = jnp.zeros_like(coeffs)

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
        
        # --- Gradient Clipping ---
        grad_norm = jnp.linalg.norm(grads)
        if grad_norm > MAX_GRAD_NORM:
            grads = grads * (MAX_GRAD_NORM / grad_norm)
            
        # --- Tangent Projection ---
        dot_prods = jnp.sum(grads * coeffs, axis=1, keepdims=True)
        grads_tangent = grads - dot_prods * coeffs
        
        # --- SGD Update with Momentum ---
        velocity = MOMENTUM * velocity + grads_tangent
        coeffs = coeffs - LEARNING_RATE * velocity
        
        # Renormalize to stay on sphere
        coeffs = normalize_coeffs(coeffs)
        
        epoch_time = time.time() - start_time
        print(f"Step {step+1:4d} | Loss: {loss_val:.6f} | GNorm: {grad_norm:.4f} | Lag: {lag_loss:.6f} | Time: {epoch_time:.2f}s")

    print("\nOptimization Complete.")
    print("Final Coefficients:")
    print(coeffs)

if __name__ == "__main__":
    main()
