import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
from functools import partial

from find_smooth_submanifold import refine_point_iterative, normalize_coeffs
from slag_condition import (
    vmap_compute_affine_jacobian, 
    vmap_compute_restriction,
    compute_kahler_form_unrestricted, 
    compute_holomorphic_form_restricted,
    compute_special_condition_fitness_smooth
)
from helper import convert_real_to_complex_batch, determine_patches_batch

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
PSI = 0
METRIC = 'k4_fermat'
NEWTON_STEPS = 40
EPSILON = 1e-4

# -----------------------------------------------------------------------------
# LOSS FUNCTION
# -----------------------------------------------------------------------------
def compute_loss_simple(
    coeffs: jnp.ndarray,
    fixed_points_real: jnp.ndarray,
    psi: jnp.ndarray,
    metric: str
) -> jnp.float32:
    
    # 1. Refine
    refine_fn = partial(
        refine_point_iterative,
        coeffs=coeffs,
        psi=psi,
        n_steps=NEWTON_STEPS
    )
    min_set_real = jax.vmap(refine_fn)(fixed_points_real)

    # 2. Compute Fitness
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
    
    # Simple mean for gradient checking (no sorting/cutting to avoid discontinuities)
    lagrangian_loss = jnp.mean(norms_normalized)
    
    return lagrangian_loss

loss_and_grad = jax.jit(jax.value_and_grad(compute_loss_simple), static_argnames=('metric',))

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("--- Gradient Check ---")
    
    # Generate random points
    key = jax.random.PRNGKey(0)
    random_complex = jax.random.normal(key, (10, 5), dtype=jnp.complex64)
    random_complex = random_complex / jnp.linalg.norm(random_complex, axis=1, keepdims=True)
    points_real = jnp.concatenate([jnp.real(random_complex), jnp.imag(random_complex)], axis=1)

    # Init Coeffs
    key = jax.random.PRNGKey(42)
    coeffs = jax.random.uniform(key, (3, 25), minval=-1.0, maxval=1.0)
    coeffs = normalize_coeffs(coeffs)
    
    print("Computing JAX Gradient...")
    loss_val, grads = loss_and_grad(coeffs, points_real, PSI, METRIC)
    print(f"Loss: {loss_val}")
    
    # Check a specific coefficient, e.g., coeffs[0, 0]
    idx = (0, 0)
    analytical_grad = grads[idx]
    
    print(f"\nChecking Gradient for coeffs{idx}...")
    print(f"Analytical (JAX) Grad: {analytical_grad}")
    
    # Perturb +
    coeffs_plus = coeffs.at[idx].add(EPSILON)
    loss_plus = compute_loss_simple(coeffs_plus, points_real, PSI, METRIC)
    
    # Perturb -
    coeffs_minus = coeffs.at[idx].add(-EPSILON)
    loss_minus = compute_loss_simple(coeffs_minus, points_real, PSI, METRIC)
    
    numerical_grad = (loss_plus - loss_minus) / (2 * EPSILON)
    print(f"Numerical Grad:        {numerical_grad}")
    
    diff = jnp.abs(analytical_grad - numerical_grad)
    print(f"Difference:            {diff}")
    
    if diff < 1e-5:
        print("\nSUCCESS: Gradient matches!")
    else:
        print("\nFAILURE: Gradient mismatch!")
        print(f"Relative Error: {diff / (jnp.abs(numerical_grad) + 1e-9)}")

if __name__ == "__main__":
    main()
