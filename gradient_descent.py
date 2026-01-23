import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
import argparse
import os
from functools import partial

from find_smooth_submanifold import filter_and_refine, normalize_coeffs
from slag_condition import compute_combined_fitness, compute_special_condition_fitness_smooth
from helper import canonicalize_coeffs

# Ensure 64-bit precision if needed, though usually float32 is faster for ML
# jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
PSI = 0
CYPOINTSFILE = f'/projects/ruehlehet/yidi/sLag/data_psi/1mil_patch_all_psi{PSI}_seed1024.pkl'
METRIC = 'k4_fermat'

# Optimization Parameters
LEARNING_RATE = 0.01
NUM_STEPS = 200
MINSET_SIZE = 10000
NEWTON_STEPS = 40  # Reduced for speed during gradient steps? Or keep high for accuracy.

# -----------------------------------------------------------------------------
# 2. LOSS FUNCTION
# -----------------------------------------------------------------------------

def compute_loss(
    coeffs: jnp.ndarray,
    points_real: jnp.ndarray,
    psi: jnp.ndarray,
    k: int,
    n_refine_steps: int,
    metric: str
) -> jnp.float32:
    """
    Differentiable loss function.
    We want to MAXIMIZE fitness, so we MINIMIZE loss = -fitness.
    """
    
    # 1. Newton's Method Refinement (Differentiable)
    # Note: The selection of 'initial' points inside filter_and_refine is non-differentiable
    # but JAX effectively treats the indices as constants for the gradient calculation.
    # We disable the "repulsion" step for the gradient calculation to keep it simpler 
    # and purely focused on the manifold shape, though repulsion is also differentiable.
    min_set_real, _, newton_check_pass = filter_and_refine(
        points_real, coeffs, psi, k, n_refine_steps, 
        filter_newton=True,
        n_repulsion_steps=0 # Disable repulsion for pure gradient descent on shape
    )

    # 2. Compute Fitness
    # We use a custom version of compute_combined_fitness logic here to use the smooth metric
    
    # Re-implementing parts of compute_combined_fitness to use the smooth function
    from helper import convert_real_to_complex_batch, determine_patches_batch
    from find_smooth_submanifold import vmap_compute_affine_jacobian, vmap_compute_restriction
    from slag_condition import compute_kahler_form_unrestricted, compute_lagrangian_condition_fitness, compute_holomorphic_form_restricted

    min_set = convert_real_to_complex_batch(min_set_real)
    patch_indices = determine_patches_batch(min_set) 

    jacobians = vmap_compute_affine_jacobian(min_set_real, patch_indices, coeffs, psi)
    restrictions = vmap_compute_restriction(jacobians)

    kahler_form_unrestricted = compute_kahler_form_unrestricted(min_set, patch_indices, metric=metric)

    # --- MODIFIED LOSS CALCULATION (Aiming for 0) ---

    # 1. Lagrangian Loss (Kahler Form condition)
    # Re-implementing the core logic of compute_lagrangian_condition_fitness 
    # but extracting the raw loss instead of the exponentiated fitness.
    kahler_form_restricted = jnp.einsum('nij,nik,njl->nkl', kahler_form_unrestricted, restrictions, restrictions)
    frobenius_norms = jnp.linalg.norm(kahler_form_restricted, axis=(1, 2))
    normalization_factor = jnp.linalg.norm(kahler_form_unrestricted, axis=(1, 2))
    
    # Avoid division by zero if normalization factor is tiny (unlikely but safe)
    norms_normalized = frobenius_norms / (normalization_factor + 1e-9)
    
    # We prune the top 1% to be robust against outliers (similar to original logic)
    # but for gradient descent, sometimes smooth mean is better. 
    # Let's keep the sort/prune for consistency with the GA metric.
    sorted_norms = jnp.sort(norms_normalized)
    # Cut top 1%
    cutoff_index = int(sorted_norms.shape[0] * 0.99)
    norms_cut = sorted_norms[:cutoff_index]
    
    lagrangian_loss = jnp.mean(norms_cut)

    # 2. Special Loss (Phase condition)
    # Use the smooth phase calculation
    phases = compute_holomorphic_form_restricted(
        min_set, patch_indices, restrictions, phase_only=True
    )
    
    # Order Parameter is in [0, 1]. 1 means concentrated.
    order_parameter = compute_special_condition_fitness_smooth(phases)
    special_loss = 1.0 - order_parameter
    
    # 3. Total Loss
    # We sum them. You might need to weight them if they have vastly different scales.
    # Typically lagrangian_loss is small (< 0.1) and special_loss can be large (up to 1.0).
    # Let's try a 1:1 ratio first, or maybe weight lagrangian higher.
    # For now, simple sum.
    total_loss = lagrangian_loss + special_loss
    
    return total_loss, (lagrangian_loss, special_loss)

# Create a value_and_grad function
# has_aux=True allows us to return auxiliary data (individual fitness components)
loss_value_and_grad = jax.jit(jax.value_and_grad(compute_loss, argnums=0, has_aux=True), static_argnames=('k', 'n_refine_steps', 'metric'))


# -----------------------------------------------------------------------------
# 3. MAIN LOOP
# -----------------------------------------------------------------------------

def main():
    print("--- Gradient Descent Optimization ---")
    
    # --- Load Data ---
    # In a real run, this loads from the cluster path.
    # For local testing without the file, we might need a mock, but I will assume
    # the user is running this where the file exists or has mocked it.
    try:
        with open(CYPOINTSFILE, 'rb') as f:
            points_real = np.asarray(pickle.load(f))
        points_real = np.concatenate([np.real(points_real), np.imag(points_real)], axis=1)
        points_real = jax.device_put(jnp.asarray(points_real))
        print(f"Loaded {len(points_real)} points.")
    except FileNotFoundError:
        print(f"Warning: Data file {CYPOINTSFILE} not found. Generating random points for testing.")
        key = jax.random.PRNGKey(0)
        # Generate random complex points on CP4 (rough approx)
        random_complex = jax.random.normal(key, (100000, 5), dtype=jnp.complex64)
        random_complex = random_complex / jnp.linalg.norm(random_complex, axis=1, keepdims=True)
        points_real = jnp.concatenate([jnp.real(random_complex), jnp.imag(random_complex)], axis=1)

    # --- Initialize Coefficients ---
    key = jax.random.PRNGKey(42)
    # Start with random coefficients
    coeffs = jax.random.uniform(key, (3, 25), minval=-1.0, maxval=1.0)
    coeffs = normalize_coeffs(canonicalize_coeffs(coeffs))
    
    print("Initial coefficients set.")
    
    # --- Optimization Loop ---
    print(f"Starting GD for {NUM_STEPS} steps with LR={LEARNING_RATE}...")
    
    for step in range(NUM_STEPS):
        start_time = time.time()
        
        # Calculate gradients
        # Note: We re-scan the 'initial points' every step implicitly because 
        # filter_and_refine is called inside compute_loss.
        (loss_val, (lag_fit, spec_fit)), grads = loss_value_and_grad(
            coeffs, points_real, PSI, MINSET_SIZE, NEWTON_STEPS, METRIC
        )
        
        # Gradient Descent Update
        # coeffs = coeffs - LEARNING_RATE * grads
        
        # Normalized Gradient Descent (often more stable for coefficients on a sphere)
        # We can project the gradient or just normalize after.
        coeffs = coeffs - LEARNING_RATE * grads
        coeffs = normalize_coeffs(canonicalize_coeffs(coeffs))
        
        epoch_time = time.time() - start_time
        
        print(f"Step {step+1:3d} | Loss: {loss_val:.6f} | Lag: {lag_fit:.4f} | Spec: {spec_fit:.4f} | Time: {epoch_time:.2f}s")
        
        # Optional: Save checkpoints or check for convergence
        if (step + 1) % 10 == 0:
            # Save current best
            pass

    print("\nOptimization Complete.")
    print("Final Coefficients:")
    print(coeffs)

if __name__ == "__main__":
    main()
