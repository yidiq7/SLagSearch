import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
import argparse
import os
from functools import partial
import optax # Restore Optax

from find_smooth_submanifold import compute_distances_batched, evaluate_equations_single_point, compute_affine_jacobian, PATCH_ACTIVE_INDICES, filter_and_refine, normalize_coeffs

from helper import canonicalize_coeffs, convert_real_to_complex_single, convert_complex_to_real_single, determine_patch_and_rescale_single, convert_real_to_complex_batch, determine_patches_batch, format_array_with_commas

from slag_condition import (
    vmap_compute_affine_jacobian, 
    vmap_compute_restriction,
    compute_kahler_form_unrestricted, 
    compute_holomorphic_form_restricted,
    compute_special_condition_fitness_smooth
    )

 
# Enable 64-bit precision (Critical for stability)
jax.config.update("jax_enable_x64", True)

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
    min_set_real: jnp.ndarray,
    psi: jnp.ndarray,
    n_refine_steps: int,
    metric: str
) -> tuple[jnp.float32, tuple[jnp.float32, jnp.float32]]:
    
    # Use the SIMPLIFIED solver
    refine_fn = partial(refine_point_iterative_simple, coeffs=coeffs, psi=psi, n_steps=n_refine_steps)
    min_set_real = jax.vmap(refine_fn)(min_set_real)

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
        min_set, patch_indices, psi, restrictions, phase_only=True
    )
    order_parameter = compute_special_condition_fitness_smooth(phases)
    special_loss = 1.0 - order_parameter
    
    # Combine both
    total_loss = lagrangian_loss
    #total_loss = special_loss
    #total_loss = lagrangian_loss + special_loss
    
    return total_loss, (lagrangian_loss, special_loss)

loss_value_and_grad = jax.jit(jax.value_and_grad(compute_loss_on_fixed_points, argnums=0, has_aux=True), static_argnames=('n_refine_steps', 'metric'))

# -----------------------------------------------------------------------------
# 4. MAIN LOOP (Adam)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gradient Descent for sLag Search")
    parser.add_argument("--psi", type=int, default=0, help="PSI parameter")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimization steps")
    parser.add_argument("--job_id", type=str, default="0", help="Job ID for saving results")
    args = parser.parse_args()

    PSI = args.psi
    LEARNING_RATE = args.lr
    NUM_STEPS = args.steps
    METRIC = 'k4_fermat'
    MINSET_SIZE = 10000
    NEWTON_STEPS = 50
    MINE_INTERVAL = 10 # Disable re-mining for this test

 
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

    # Init Coeffs (Explicit Initialization)
    # key = jax.random.PRNGKey(int(args.job_id) if args.job_id.isdigit() else 42) 
    # coeffs = jax.random.uniform(key, (3, 25), minval=-1.0, maxval=1.0)
    
    coeffs_init = np.array([
        [-0.0367027148604393, 0.0025424568448215723, -0.21919819712638855, 0.459602952003479, -0.14038687944412231, 0.17836026847362518, -0.02632879465818405, 0.1542496383190155, -0.21669824421405792, 0.08787675201892853, -0.02216939628124237, 0.4584207534790039, 0.18183615803718567, 0.006892359349876642, 0.09927520900964737, -0.019934242591261864, -0.16006404161453247, -0.1488436758518219, 0.4862450659275055, 0.07581081986427307, -0.1970602571964264, 0.04495194926857948, 0.09229589253664017, 0.15134599804878235, 0.005224315449595451],
        [0.42350825667381287, 0.16512976586818695, -0.009587904438376427, 0.020616931840777397, -0.10060423612594604, 0.11577785015106201, 0.3959497809410095, -0.23205921053886414, -0.007092039100825787, 0.1421680897474289, -0.033782344311475754, 0.15470725297927856, -0.05985529348254204, -0.11284781247377396, 0.41878288984298706, 0.012480814009904861, 0.09610676020383835, 0.09385758638381958, 0.1206778883934021, -0.05080771818757057, -0.5067381262779236, -0.14283324778079987, -0.01962202973663807, -0.10288462787866592, 0.02445816993713379],
        [0.3107169568538666, -0.16336588561534882, 0.13979989290237427, -0.06237269937992096, 0.20755282044410706, -0.18239142000675201, 0.2927098274230957, 0.17870883643627167, 0.15436840057373047, -0.20919054746627808, -0.014542280696332455, 0.08693742007017136, -0.039950814098119736, 0.14380718767642975, 0.2939481735229492, -0.007748228497803211, -0.04357427731156349, -0.07980721443891525, 0.08186690509319305, -0.03721674904227257, 0.6561188101768494, 0.1590929478406906, -0.06476886570453644, -0.03910057991743088, 0.025055011734366417]
    ])

    coeffs = jnp.array(coeffs_init, dtype=jnp.float64) # Ensure x64
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
            #mine_start = time.time()
            #active_indices = mine_indices(coeffs, points_real, PSI, MINSET_SIZE)
            #current_batch = points_real[active_indices]
            #mine_time = time.time() - mine_start
            #print(f"  [Mining] Selected new {MINSET_SIZE} points in {mine_time:.2f}s")
            min_set_real, _, newton_check_pass = filter_and_refine(
                points_real, coeffs, PSI, MINSET_SIZE, NEWTON_STEPS, filter_newton=True
            )
            
        # Training
        (loss_val, (lag_loss, spec_loss)), grads = loss_value_and_grad(
            coeffs, min_set_real, PSI, 10, METRIC
        )
        
        # We don't need manual clipping with Adam usually, but let's be safe against infs
        # Replace infs/nans with zero in gradients
        grads = jnp.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        
        # --- Tangent Projection (Crucial for Spherical Optimization) ---
        # Ensure gradient effectively only moves coeffs along the sphere surface
        #dot_prods = jnp.sum(grads * coeffs, axis=1, keepdims=True)
        #radial_mag = jnp.mean(jnp.abs(dot_prods))
        #grads = grads - dot_prods * coeffs
        
        updates, opt_state = optimizer.update(grads, opt_state, coeffs)
        coeffs = optax.apply_updates(coeffs, updates)
        
        # Renormalize to stay on sphere
        coeffs = normalize_coeffs(coeffs)
        
        epoch_time = time.time() - start_time
        grad_norm = jnp.linalg.norm(grads)
        print(f"Step {step+1:4d} | Total: {loss_val:.6f} | Lag: {lag_loss:.6f} | Spec: {spec_loss:.6f} | GNorm: {grad_norm:.2e} | Time: {epoch_time:.2f}s")

    print("\nOptimization Complete.")
    print("Final Coefficients:")
    print(format_array_with_commas(coeffs))

if __name__ == "__main__":
    main()
