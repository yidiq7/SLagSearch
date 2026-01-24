import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
import argparse
import os
from functools import partial
import optax  # New dependency

from find_smooth_submanifold import filter_and_refine, normalize_coeffs
from slag_condition import compute_combined_fitness, compute_special_condition_fitness_smooth
from helper import canonicalize_coeffs

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
PSI = 0
CYPOINTSFILE = f'/projects/ruehlehet/yidi/sLag/data_psi/1mil_patch_all_psi{PSI}_seed1024.pkl'
METRIC = 'k4_fermat'

# Optimization Parameters
LEARNING_RATE = 0.001 # Standard Adam LR
NUM_STEPS = 200
MINSET_SIZE = 10000
NEWTON_STEPS = 40

# -----------------------------------------------------------------------------
# 2. LOSS FUNCTION (Minimizing Error)
# -----------------------------------------------------------------------------

def compute_loss(
    coeffs: jnp.ndarray,
    points_real: jnp.ndarray,
    psi: jnp.ndarray,
    k: int,
    n_refine_steps: int,
    metric: str
) -> tuple[jnp.float32, tuple[jnp.float32, jnp.float32]]:
    """
    Differentiable loss function.
    Targets: Lagrangian Loss -> 0, Special Loss (1-OrderParam) -> 0
    """
    
    # 1. Newton's Method Refinement
    # The selection of points is effectively constant for the gradient step.
    min_set_real, _, newton_check_pass = filter_and_refine(
        points_real, coeffs, psi, k, n_refine_steps, 
        filter_newton=True,
        n_repulsion_steps=0 
    )

    # 2. Compute Fitness Components
    from helper import convert_real_to_complex_batch, determine_patches_batch
    from find_smooth_submanifold import vmap_compute_affine_jacobian, vmap_compute_restriction
    from slag_condition import compute_kahler_form_unrestricted, compute_holomorphic_form_restricted

    min_set = convert_real_to_complex_batch(min_set_real)
    patch_indices = determine_patches_batch(min_set) 

    jacobians = vmap_compute_affine_jacobian(min_set_real, patch_indices, coeffs, psi)
    restrictions = vmap_compute_restriction(jacobians)

    kahler_form_unrestricted = compute_kahler_form_unrestricted(min_set, patch_indices, metric=metric)

    # --- Lagrangian Loss Calculation ---
    kahler_form_restricted = jnp.einsum('nij,nik,njl->nkl', kahler_form_unrestricted, restrictions, restrictions)
    frobenius_norms = jnp.linalg.norm(kahler_form_restricted, axis=(1, 2))
    normalization_factor = jnp.linalg.norm(kahler_form_unrestricted, axis=(1, 2))
    
    norms_normalized = frobenius_norms / (normalization_factor + 1e-9)
    
    # Prune top 1% to be robust (matching GA logic)
    sorted_norms = jnp.sort(norms_normalized)
    cutoff_index = int(sorted_norms.shape[0] * 0.99)
    norms_cut = sorted_norms[:cutoff_index]
    
    lagrangian_loss = jnp.mean(norms_cut)

    # --- Special Loss (Phase) Calculation ---
    phases = compute_holomorphic_form_restricted(
        min_set, patch_indices, restrictions, phase_only=True
    )
    
    # Order Parameter -> 1.0 means perfect concentration.
    # Loss = 1.0 - Order Parameter
    order_parameter = compute_special_condition_fitness_smooth(phases)
    special_loss = 1.0 - order_parameter
    
    # Total Loss
    total_loss = lagrangian_loss + special_loss
    
    return total_loss, (lagrangian_loss, special_loss)

# Create a value_and_grad function
loss_value_and_grad = jax.jit(jax.value_and_grad(compute_loss, argnums=0, has_aux=True), static_argnames=('k', 'n_refine_steps', 'metric'))


# -----------------------------------------------------------------------------
# 3. MAIN LOOP
# -----------------------------------------------------------------------------

def main():
    print("--- Adam Optimization for sLag Search ---")
    
    # --- Load Data ---
    try:
        with open(CYPOINTSFILE, 'rb') as f:
            points_real = np.asarray(pickle.load(f))
        points_real = np.concatenate([np.real(points_real), np.imag(points_real)], axis=1)
        points_real = jax.device_put(jnp.asarray(points_real))
        print(f"Loaded {len(points_real)} points.")
    except FileNotFoundError:
        print(f"Warning: Data file {CYPOINTSFILE} not found. Generating random points for testing.")
        key = jax.random.PRNGKey(0)
        random_complex = jax.random.normal(key, (100000, 5), dtype=jnp.complex64)
        random_complex = random_complex / jnp.linalg.norm(random_complex, axis=1, keepdims=True)
        points_real = jnp.concatenate([jnp.real(random_complex), jnp.imag(random_complex)], axis=1)

    # --- Initialize Coefficients ---
    key = jax.random.PRNGKey(42)
    coeffs = jax.random.uniform(key, (3, 25), minval=-1.0, maxval=1.0)
    coeffs = normalize_coeffs(canonicalize_coeffs(coeffs))
    
    print("Initial coefficients set.")
    
    # --- Initialize Optimizer ---
    # We use Adam with a standard learning rate
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(coeffs)

    print(f"Starting Optimization for {NUM_STEPS} steps...")
    
    for step in range(NUM_STEPS):
        start_time = time.time()
        
        # 1. Calculate Gradients
        (loss_val, (lag_loss, spec_loss)), grads = loss_value_and_grad(
            coeffs, points_real, PSI, MINSET_SIZE, NEWTON_STEPS, METRIC
        )
        
        # 2. Update via Optax
        updates, opt_state = optimizer.update(grads, opt_state, coeffs)
        coeffs = optax.apply_updates(coeffs, updates)
        
        # 3. Manually Normalize coefficients (project back to valid space)
        # This is important because the standard Adam update might push them off the "sphere"
        coeffs = normalize_coeffs(canonicalize_coeffs(coeffs))
        
        epoch_time = time.time() - start_time
        
        print(f"Step {step+1:3d} | Total Loss: {loss_val:.6f} | Lag Loss: {lag_loss:.6f} | Spec Loss: {spec_loss:.6f} | Time: {epoch_time:.2f}s")
        
        # Save checkpoints periodically
        if (step + 1) % 50 == 0:
             pass # Add saving logic here if needed

    print("\nOptimization Complete.")
    print("Final Coefficients:")
    print(coeffs)

if __name__ == "__main__":
    main()