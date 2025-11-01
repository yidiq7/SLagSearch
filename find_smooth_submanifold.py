import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Union, Optional
from functools import partial
from collections import defaultdict
from get_restriction import compute_affine_jacobian
from helper import (evaluate_equations_single_point, convert_complex_to_real_single,
                    convert_real_to_complex_single, determine_patch_and_rescale_single)

PATCH_ACTIVE_MASKS = jnp.array([
    [False, True, True, True, True, False, True, True, True, True],   # patch=0
    [True, False, True, True, True, True, False, True, True, True],   # patch=1
    [True, True, False, True, True, True, True, False, True, True],   # patch=2
    [True, True, True, False, True, True, True, True, False, True],   # patch=3
    [True, True, True, True, False, True, False, True, True, False],   # patch=4
], dtype=bool)


PATCH_ACTIVE_INDICES = jnp.array([
    [1, 2, 3, 4, 6, 7, 8, 9],  # patch=0: skip 0,6
    [0, 2, 3, 4, 5, 7, 8, 9],  # patch=1: skip 1,6
    [0, 1, 3, 4, 5, 6, 8, 9],  # patch=2: skip 2,6
    [0, 1, 2, 4, 5, 6, 7, 9],  # patch=3: skip 3,6
    [0, 1, 2, 3, 5, 6, 7, 8],  # patch=4: skip 4,6
], dtype=jnp.int32)

# Enable 64-bit precision for complex numbers
#jax.config.update("jax_enable_x64", True)
def combine_to_complex_equations(labels, coeffs):
    """
    Convert equations from Re/Im basis to zi*zjbar basis using:
    Im(zi*zjbar) = (zi*zjbar - zj*zibar)/(2i)
    Re(zi*zjbar) = (zi*zjbar + zj*zibar)/2
    
    Parameters:
    labels: array of strings like 'Im(z0*z1bar)', 'Re(z0*z1bar)'
    coeffs: 3x25 array of coefficients
    
    Returns:
    List of 3 strings representing the equations in terms of zi*zjbar
    """
    equations = []
    
    for eq_idx in range(coeffs.shape[0]):
        # Dictionary to store coefficients for each zi*zjbar term
        terms = defaultdict(complex)
        
        # Process each label and its coefficient
        for label_idx, label in enumerate(labels):
            coeff = coeffs[eq_idx, label_idx]
            
            # Skip if coefficient is very small
            if abs(coeff) < 1e-10:
                continue
            
            # Extract the zi and zj indices
            if label.startswith('Im(') and label.endswith(')'):
                # Extract 'zi*zjbar' from 'Im(zi*zjbar)'
                inner = label[3:-1]  # e.g., 'z0*z1bar'
                parts = inner.split('*')
                zi = parts[0]  # e.g., 'z0'
                zj = parts[1].replace('bar', '')  # e.g., 'z1'
                
                # Im(zi*zjbar) = (zi*zjbar - zj*zibar)/(2i)
                # Coefficient for zi*zjbar: coeff/(2i) = -coeff*i/2
                # Coefficient for zj*zibar: -coeff/(2i) = coeff*i/2
                
                terms[f"{zi}*{zj}bar"] += complex(0, -coeff/2)  # -i*coeff/2
                terms[f"{zj}*{zi}bar"] += complex(0, coeff/2)   # i*coeff/2
                
            elif label.startswith('Re(') and label.endswith(')'):
                # Extract 'zi*zjbar' from 'Re(zi*zjbar)'
                inner = label[3:-1]  # e.g., 'z0*z1bar'
                parts = inner.split('*')
                zi = parts[0]  # e.g., 'z0'
                zj = parts[1].replace('bar', '')  # e.g., 'z1'
                
                # Re(zi*zjbar) = (zi*zjbar + zj*zibar)/2
                # Coefficient for both zi*zjbar and zj*zibar: coeff/2
                
                terms[f"{zi}*{zj}bar"] += complex(coeff/2, 0)
                terms[f"{zj}*{zi}bar"] += complex(coeff/2, 0)
        
        # Build the equation string
        equation_parts = []
        for term in sorted(terms.keys()):
            coeff_complex = terms[term]
            
            # Skip if coefficient is essentially zero
            if abs(coeff_complex) < 1e-10:
                continue
            
            # Format the complex coefficient
            real_part = coeff_complex.real
            imag_part = coeff_complex.imag
            
            if abs(imag_part) < 1e-10:
                # Only real part
                coeff_str = f"{real_part:.6f}"
            elif abs(real_part) < 1e-10:
                # Only imaginary part
                if abs(imag_part - 1) < 1e-10:
                    coeff_str = "i"
                elif abs(imag_part + 1) < 1e-10:
                    coeff_str = "-i"
                else:
                    coeff_str = f"{imag_part:.6f}i"
            else:
                # Both real and imaginary parts
                if imag_part >= 0:
                    coeff_str = f"({real_part:.6f}+{imag_part:.6f}i)"
                else:
                    coeff_str = f"({real_part:.6f}{imag_part:.6f}i)"
            
            # Add to equation
            if equation_parts:
                if coeff_str.startswith('-'):
                    equation_parts.append(f" {coeff_str}*{term}")
                else:
                    equation_parts.append(f" + {coeff_str}*{term}")
            else:
                equation_parts.append(f"{coeff_str}*{term}")
        
        equation = "".join(equation_parts) + " = 0"
        equations.append(equation)
    
    return equations


@jax.jit
def generate_basis(points: jnp.ndarray) -> jnp.ndarray:
    """
    Generate basis functions from points on Fermat quintic.
    
    Args:
        points: (N, 5) complex array of points on the quintic
        
    Returns:
        basis: (N, 25) real array of basis functions
               First 10 are Im(zi*zjbar) for i<j, next 15 are Re(zi*zjbar) for i<=j
    """
    N = points.shape[0]
    
    # Create all pairwise products zi * zj_bar using broadcasting
    # Shape: (N, 5, 5)
    zi = points[:, :, None]  # (N, 5, 1)
    zj_bar = jnp.conj(points[:, None, :])  # (N, 1, 5)
    products = zi * zj_bar  # (N, 5, 5)
    
    # Extract upper triangular indices for imaginary parts (i < j)
    # This gives us 10 unique imaginary parts
    triu_indices_imag = jnp.triu_indices(5, k=1)
    imag_basis = jnp.imag(products[:, triu_indices_imag[0], triu_indices_imag[1]])  # (N, 10)
    
    # Extract upper triangular indices including diagonal for real parts (i <= j)
    # This gives us 15 unique real parts
    triu_indices_real = jnp.triu_indices(5, k=0)
    real_basis = jnp.real(products[:, triu_indices_real[0], triu_indices_real[1]])  # (N, 15)
    
    # Concatenate to form complete basis
    return jnp.concatenate([imag_basis, real_basis], axis=1)  # (N, 25)

@jax.jit 
def normalize_coeffs(coeffs: jnp.ndarray) -> jnp.ndarray:
    # We normalize on the complex basis zizjbar instead of the real and imaginary
    # parts. So we rescale the real part of zizibar by 1/sqrt(2) to get the correct
    # normalization since they are only counted once instead of twice compared to
    # the upper triangular terms.
    zzbar_indices = jnp.array([10, 15, 19, 22, 24])
    weights = jnp.ones((3, 25))
    weights = weights.at[:,zzbar_indices].divide(jnp.sqrt(2.0))
    norms = jnp.linalg.norm(weights*coeffs, axis=1, keepdims=True)
    coeffs_normalized = coeffs / norms
    return coeffs_normalized


def get_basis_labels():
    """Get human-readable labels for basis functions."""
    labels = []
    
    # Imaginary parts for i < j
    for i in range(5):
        for j in range(i+1, 5):
            labels.append(f"Im(z{i}*z{j}bar)")
    
    # Real parts for i <= j
    for i in range(5):
        for j in range(i, 5):
            labels.append(f"Re(z{i}*z{j}bar)")
    
    return labels


# ------------------------------------------------------------------------------
# New loss function
def approx_distance_newton_step(
    p_10d: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray
) -> float:
    """
    Computes the norm of a Newton step with automatic patch detection.
    
    Args:
        p_10d: A single point in (10,) real representation
        coeffs: A (3, 25) coefficient array
        psi: Complex parameter
        
    Returns:
        The norm of the Newton step (distance to manifold approximation)
    """

    # Compute Newton step in this patch
    f_vec = evaluate_equations_single_point(p_10d, coeffs, psi)
    J = jax.jacobian(evaluate_equations_single_point, argnums=0)(p_10d, coeffs, psi)
    JJT = J @ J.T + 1e-8 * jnp.eye(J.shape[0])
    w = jnp.linalg.solve(JJT, -f_vec)
    delta_p_active = J.T @ w
    
    return jnp.linalg.norm(delta_p_active)


def refine_point_iterative(
    p_10d_initial: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    n_steps: int
) -> jnp.ndarray:
    """
    Refines a point using Newton's method with automatic patch handling.
    
    At each iteration, the point is rescaled to its appropriate patch after 
    computing the Newton step. This ensures numerical stability even as the
    point moves through projective space.
    
    Args:
        p_10d_initial: Initial point in (10,) real representation
        coeffs: A (3, 25) coefficient array
        psi: Complex parameter
        n_steps: Number of Newton iterations
        
    Returns:
        Refined point in (10,) real representation
    """

    p_complex_init = convert_real_to_complex_single(p_10d_initial)
    _, patch_index_init = determine_patch_and_rescale_single(p_complex_init)
    init_state = (p_complex_init, patch_index_init)

    def body_fn(i, state):
        p_10d, patch_index = state
        active_indices = PATCH_ACTIVE_INDICES[patch_index]

        # Compute Newton step
        f_vec = evaluate_equations_single_point(p_10d, coeffs, psi)
        #J = jax.jacobian(evaluate_equations_single_point, argnums=0)(p_10d, coeffs, psi)
        J = compute_affine_jacobian(p_10d, patch_index, coeffs, psi)
        JJT = J @ J.T + 1e-8 * jnp.eye(J.shape[0])
        w = jnp.linalg.solve(JJT, -f_vec)
        delta_p_active = J.T @ w

        #p_10d += delta_p_active
        p_10d = p_10d.at[active_indices].add(delta_p_active)

        p_complex = convert_real_to_complex_single(p_10d)
        p_complex_rescaled, patch_index = determine_patch_and_rescale_single(p_complex)
        p_10d_rescaled = convert_complex_to_real_single(p_complex_rescaled) 

        # The updated point is returned in rescaled form (largest coord = 1)
        # This is correct for the next iteration
        return (p_10d_rescaled, patch_index)
    
    p_10d_final, _ = jax.lax.fori_loop(0, n_steps, body_fn, init_state)
    return p_10d_final


def compute_distances_batched(
    points: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes distances for a batch of points with automatic patch handling.
    
    Args:
        points: An (N, 10) real array
        coeffs: A (3, 25) coefficient array
        psi: Complex parameter
        
    Returns:
        An (N,) array of distances
    """
    dist_partial = partial(
        approx_distance_newton_step,
        coeffs=coeffs,
        psi=psi
    )
    
    return jax.jit(jax.vmap(dist_partial))(points)


@partial(jax.jit, static_argnames=('constant_coord',))
def _project_forces_to_tangent_space(
    points: jnp.ndarray,
    forces: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    constant_coord: int
) -> jnp.ndarray:
    """
    Projects a batch of 10D force vectors onto the tangent spaces at each point.
    This is a helper function for the repulsion algorithm.
    """
    # 1. Get Jacobians for all points in the batch using vmap
    batch_jacobian_fn = jax.vmap(compute_affine_jacobian, in_axes=(0, 0, None, None))
    jacobians = batch_jacobian_fn(points, coeffs, psi, constant_coord)  # Shape: (k, 5, 8)

    # 2. Extract the 8 active components from the 10D force vectors
    active_indices = jnp.concatenate([
        jnp.arange(0, constant_coord),
        jnp.arange(constant_coord + 1, 5),
        jnp.arange(5, constant_coord + 5),
        jnp.arange(constant_coord + 6, 10)
    ])
    forces_active = forces[:, active_indices]  # Shape: (k, 8)

    # 3. Project forces onto the tangent space (v_tangent = v - v_normal)
    # --- FIX STARTS HERE ---
    # The original string 'kmi,km->ki' had a label mismatch.
    # This corrected string 'kmi,ki->km' correctly performs the batch multiplication
    # of the Jacobians (k, 5, 8) with forces_active (k, 8).
    J_forces_active = jnp.einsum('kmi,ki->km', jacobians, forces_active)  # Shape: (k, 5)
    # --- FIX ENDS HERE ---
    
    JJT = jnp.einsum('kmi,kni->kmn', jacobians, jacobians) + 1e-6 * jnp.eye(5)

    w = jnp.linalg.solve(JJT, J_forces_active[..., None]).squeeze(axis=-1)
    # This calculation for the normal component is correct.
    forces_normal_active = jnp.einsum('kmi,km->ki', jacobians, w)  # Shape: (k, 8)
    forces_tangent_active = forces_active - forces_normal_active

    # 4. Embed the 8D tangent forces back into 10D vectors
    return jnp.zeros_like(forces).at[:, active_indices].set(forces_tangent_active)


@partial(jax.jit, static_argnames=('k', 'n_refine_steps', 'filter_newton', 'n_repulsion_steps'))
def filter_and_refine(
    points: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: Optional[jnp.ndarray] = None,
    k: int = 10000,
    n_refine_steps: int = 20,
    filter_newton: bool = False,
    n_repulsion_steps: int = 20,
    repulsion_strength: Optional[float] = None,
    repulsion_radius: Optional[float] = None
) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
    """
    Filters and refines points with automatic patch handling.
    
    This is a drop-in replacement for filter_and_refine that works across
    all coordinate patches. Points are automatically rescaled to their
    appropriate patch at each step.
    
    Args:
        points: An (N, 10) real array of initial points
        coeffs: A (3, 25) coefficient array
        psi: Complex parameter
        k: Number of points to keep
        n_refine_steps: Number of Newton iterations per refinement
        filter_newton: If True, check convergence and return flag
        n_repulsion_steps: Number of repulsion iterations
        repulsion_strength: Strength of repulsion force
        repulsion_radius: Radius for repulsion interaction
        
    Returns:
        final_points: A (k, 10) array of refined points
        final_distances: A (k,) array of distances to manifold
        newton_check_pass: Boolean indicating if Newton's method converged well
    """
    if psi is None:
        psi = jnp.complex64(0)
    
    # --- STEP 1: Initial filtering and refinement ---
    refine_fn = partial(
        refine_point_iterative,
        coeffs=coeffs,
        psi=psi,
        n_steps=n_refine_steps
    )
    refine_batch = jax.vmap(refine_fn)
    
    # Compute initial distances
    all_distances = compute_distances_batched(points, coeffs, psi)
    
    # Select best 2k points
    best_2k_indices = jnp.argsort(all_distances)[:2*k]
    top_2k_points = points[best_2k_indices]
    
    # Refine these points
    refined_points_10d = refine_batch(top_2k_points)
    distance_refined = compute_distances_batched(refined_points_10d, coeffs, psi)
    
    # Select best k points
    best_indices = jnp.argsort(distance_refined)[:k]
    top_k_points = refined_points_10d[best_indices]
    
    # Check convergence if requested
    initial_newton_check = True
    if filter_newton:
        mean_distance = jnp.nan_to_num(jnp.mean(distance_refined[best_indices]))
        initial_newton_check = (mean_distance <= 1e-4) & (mean_distance > 1e-16)
    
    # --- STEP 2: Repulsion for uniform distribution ---
    # Compute scale for repulsion parameters
    max_extent = jnp.max(top_k_points, axis=0)
    min_extent = jnp.min(top_k_points, axis=0)
    R_scale = jnp.linalg.norm(max_extent - min_extent) / 2
    R_scale = jnp.maximum(R_scale, 1e-6)
    
    if repulsion_radius is None:
        repulsion_radius = R_scale / jnp.cbrt(k)
    if repulsion_strength is None:
        repulsion_strength = 0.3 * R_scale
    
    # Since now the points are much closer to the manifold,
    # it would probably takes less refine steps.  
    reproject_fn = partial(
        refine_point_iterative,
        coeffs=coeffs,
        psi=psi,
        n_steps=10
    )
    batch_reproject = jax.vmap(reproject_fn)
    
    def repulsion_body_fn(i, points_state):
        # Compute pairwise repulsion
        diffs = points_state[:, None, :] - points_state[None, :, :]
        dists_sq = jnp.sum(diffs**2, axis=-1)
        
        inv_dist_sq = 1.0 / (dists_sq + 1e-9)
        forces = diffs * inv_dist_sq[..., None]
        
        mask = (dists_sq < repulsion_radius**2) & (dists_sq > 1e-9)
        net_force = jnp.sum(forces * mask[..., None], axis=1)
        
        # Apply force (no tangent projection for simplicity)
        tangent_norm = jnp.linalg.norm(net_force, axis=1, keepdims=True)
        unit_force = jnp.nan_to_num(net_force / (tangent_norm + 1e-9))
        moved_points = points_state + repulsion_strength * unit_force
        
        # Reproject onto manifold
        return batch_reproject(moved_points)
    
    # Run repulsion if requested
    final_points = jax.lax.cond(
        (n_repulsion_steps > 0) & initial_newton_check,
        lambda p: jax.lax.fori_loop(0, n_repulsion_steps, repulsion_body_fn, p),
        lambda p: p,
        top_k_points
    )
    
    # Compute final distances
    final_distances = compute_distances_batched(final_points, coeffs, psi)
    
    # Final convergence check
    repulsion_newton_check = True
    if filter_newton:
        mean_distance = jnp.nan_to_num(jnp.mean(final_distances))
        repulsion_newton_check = (mean_distance <= 1e-4) & (mean_distance > 1e-16)
    
    newton_check_pass = initial_newton_check & repulsion_newton_check
    
    return final_points, final_distances, newton_check_pass


@partial(jax.jit, static_argnames=('k', 'n_refine_steps', 'constant_coord', 'debug_mode'))
def filter_and_refine_old(
    points: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: Optional[jnp.ndarray] = None,
    k: int = 10000,
    n_refine_steps: int = 20,
    constant_coord: int = 0,
    debug_mode: bool = False
) -> jnp.ndarray:
    """
    Filters a large set of 10D points to find the k best, then refines them.
    """

    if psi is None:
        psi = jnp.complex64(0) 

    refine_partial = partial(refine_point_iterative, coeffs=coeffs, psi=psi, constant_coord=constant_coord, n_steps=n_refine_steps)
    refine_batch = jax.vmap(refine_partial)

    all_distances = compute_distances_batched(points, coeffs, psi, constant_coord=constant_coord)

    best_2k_indices = jnp.argsort(all_distances)[:2*k]
    #best_2k_indices = jnp.argsort(all_distances)[:k]
    top_2k_points = points[best_2k_indices]


    refined_points_10d = refine_batch(jnp.array(top_2k_points))
    distance_refined = compute_distances_batched(refined_points_10d, coeffs, psi, constant_coord=constant_coord)

    best_indices = jnp.argsort(distance_refined)[:int(1.9*k)]
    top_k_distances = distance_refined[best_indices]
    top_k_points = refined_points_10d[best_indices]
    if debug_mode:
        return top_k_points, top_k_distances
    else:
        return top_k_points

