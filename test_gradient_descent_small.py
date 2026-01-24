import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
import optax
import os

from find_smooth_submanifold import filter_and_refine, normalize_coeffs
from slag_condition import compute_combined_fitness, compute_special_condition_fitness_smooth
from helper import canonicalize_coeffs

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
PSI = 0
CYPOINTSFILE = f'/projects/ruehlehet/yidi/sLag/data_psi/1mil_patch_all_psi{PSI}_seed1024.pkl'
METRIC = 'k4_fermat'
LEARNING_RATE = 0.001
NUM_STEPS = 50
MINSET_SIZE = 10000
NEWTON_STEPS = 40

# -----------------------------------------------------------------------------
# 2. LOSS FUNCTION (Same as before)
# -----------------------------------------------------------------------------
# Importing the logic from gradient_descent.py conceptually
from gradient_descent import loss_value_and_grad

# -----------------------------------------------------------------------------
# 3. MASKED OPTIMIZATION
# -----------------------------------------------------------------------------

def main():
    print("--- Gradient Descent Test: 3x5 Subspace ---")
    
    # --- Load Data ---
    try:
        with open(CYPOINTSFILE, 'rb') as f:
            points_real = np.asarray(pickle.load(f))
        points_real = np.concatenate([np.real(points_real), np.imag(points_real)], axis=1)
        points_real = jax.device_put(jnp.asarray(points_real))
        print(f"Loaded {len(points_real)} points.")
    except FileNotFoundError:
        print(f"Warning: Data file not found. Using random points.")
        key = jax.random.PRNGKey(0)
        random_complex = jax.random.normal(key, (50000, 5), dtype=jnp.complex64)
        random_complex = random_complex / jnp.linalg.norm(random_complex, axis=1, keepdims=True)
        points_real = jnp.concatenate([jnp.real(random_complex), jnp.imag(random_complex)], axis=1)

    # --- Initialize 3x5 Coefficients ---
    key = jax.random.PRNGKey(99)
    # Initialize full matrix with zeros
    coeffs_full = jnp.zeros((3, 25))
    # Initialize random 3x5 block
    coeffs_small = jax.random.uniform(key, (3, 5), minval=-1.0, maxval=1.0)
    
    # Place 3x5 block into the full matrix
    coeffs = coeffs_full.at[:, :5].set(coeffs_small)
    coeffs = normalize_coeffs(coeffs)
    
    # Create Mask
    mask = jnp.zeros((3, 25))
    mask = mask.at[:, :5].set(1.0)
    
    print("Initial 3x5 coefficients set (rest are zero).")
    
    # --- Initialize Optimizer ---
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(coeffs)

    print(f"Starting Test for {NUM_STEPS} steps...")
    
    for step in range(NUM_STEPS):
        start_time = time.time()
        
        # 1. Calculate Gradients
        (loss_val, (lag_loss, spec_loss)), grads = loss_value_and_grad(
            coeffs, points_real, PSI, MINSET_SIZE, NEWTON_STEPS, METRIC
        )
        
        # 2. Apply Mask to Gradients (Force zeros in unused columns)
        grads = grads * mask
        
        # 3. Update
        updates, opt_state = optimizer.update(grads, opt_state, coeffs)
        coeffs = optax.apply_updates(coeffs, updates)
        
        # 4. Normalize
        coeffs = normalize_coeffs(coeffs)
        
        # 5. Verify Zeros (Sanity Check)
        zero_norm = jnp.linalg.norm(coeffs[:, 5:])
        if zero_norm > 1e-6:
             print(f"WARNING: Non-zero values leaked into columns 5-24! Norm: {zero_norm}")

        epoch_time = time.time() - start_time
        print(f"Step {step+1:3d} | Total Loss: {loss_val:.6f} | Lag Loss: {lag_loss:.6f} | Spec Loss: {spec_loss:.6f} | Time: {epoch_time:.2f}s")

    print("\nTest Complete.")
    print("Final 3x5 Block:")
    print(coeffs[:, :5])

if __name__ == "__main__":
    main()
