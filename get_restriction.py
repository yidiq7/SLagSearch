from typing import Optional, List
import jax
import jax.numpy as jnp
import itertools
from functools import partial
from helper import evaluate_equations_single_point


def select_active_jacobian(single_patch_index, full_jacobian):
  """
  Selects a fixed number (8) of active columns for a single patch_index.
  """
  
  M = 10  # Total number of columns
  num_active = 8 # The fixed number of columns to select
  all_indices = jnp.arange(M)

  # 1. Create the boolean mask
  exclude_mask = (all_indices == single_patch_index) | (all_indices == (single_patch_index + 5))
  active_mask = ~exclude_mask # Shape (10,)

  # 2. Convert boolean mask to fixed-size integer indices
  #    This is the "mask-to-indices" pattern.
  
  # Set "inactive" indices (where mask is False) to a large value (M)
  # Set "active" indices (where mask is True) to their own index
  masked_indices = jnp.where(active_mask, all_indices, M)

  # 3. Sort the indices. The M-2 active indices (0-9) will come
  #    first, followed by the "inactive" indices (value M).
  sorted_indices = jax.lax.sort(masked_indices) # Shape (10,)
  
  # 4. Slice the *indices* to get a fixed-size array
  #    We take the first `num_active` (8) indices.
  active_indices = sorted_indices[:num_active] # Shape (8,)

  # 5. Use these *integer* indices to select columns
  #    This is now jit- and vmap-safe because active_indices
  #    is an integer array with a concrete shape (8,).
  return full_jacobian[:, active_indices] # Returns shape (K, 8)


def compute_affine_jacobian(p_10d: jnp.ndarray, patch_index: int, coeffs: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the 5x8 Jacobian of the full system (Quintic + 3 custom equations)
    with respect to the 8 active affine real coordinates.

    Args:
        p_10d: A single point, shape (10,). The first 5 elements are x-coords, the next 5 are y-coords.
        coeffs: The coefficient matrix for one individual, shape (3, 25).
        patch_indices: The index of the z-coordinate held constant (0 to 4).

    Returns:
        The Jacobian matrix, shape (5, 8).
    """

    # Use jax.jacobian to automatically compute the derivative.
    # argnums=0 means "differentiate with respect to the first argument (point_10d)".
    # This gives the full 5x10 jacobian (5 equations, 10 real coordinates).
    #full_jacobian = jax.jacobian(evaluate_all_five_equations, argnums=0)(p_10d, coeffs)
    full_jacobian = jax.jacobian(evaluate_equations_single_point, argnums=0)(p_10d, coeffs, psi)

    # Now, select the 8 columns corresponding to the active affine coordinates.
    # The dropped coordinates are x_k and y_k, where k = constant_coord.
    # Their indices in the 10D vector are `constant_coord` and `constant_coord + 5`.
    affine_jacobian = select_active_jacobian(patch_index, full_jacobian)
    
    return affine_jacobian 

# Pre-compute all 56 combinations of 5 column indices from 0 to 7.
# This array is static and will not be recomputed during JAX transformations (like vmap).
all_column_indices_range = jnp.arange(8)
combinations_list = list(itertools.combinations(all_column_indices_range.tolist(), 5))
ALL_COMBINATIONS = jnp.array(combinations_list, dtype=jnp.int32) # Shape (56, 5)

def compute_restriction(
    eqlist_jacobian: jnp.ndarray, return_margin: bool = False
) -> jnp.ndarray:
    """
    Processes a 5x8 JAX array according to the specified steps:
    1. Finds the combination of 5 columns that forms a 5x5 submatrix
       with the largest |determinant|.
    2. Creates an 8x8 identity matrix and replaces rows corresponding
       to the chosen column indices with the rows of the input matrix.
    3. Computes the inverse of this new 8x8 matrix.
    4. Deletes the columns from the inverted matrix that correspond
       to the chosen column indices, resulting in an 8x3 array (a basis of
       T_pL spanning ker(J)).
    5. Conormally co-orients that basis: negates one column when needed so the
       equation-adapted frame det[J^T | basis] is positive (see the inline
       comment). This pins the +-1 orientation gauge of T_pL geometrically, so
       the downstream Omega|_L phase is well defined on the transverse part of L.

    This function is designed to be JAX-transformable and efficient,
    especially when used with `jax.vmap`.

    Args:
        eqlist_jacobian: A 5x8 JAX array of floating-point numbers.
        return_margin: if True, also return the scale-free co-orientation
            margin |det of column-normalised [J^T | basis]| in [0, 1] (small
            => near-degenerate => unstable sign => L not cleanly transverse
            there).

    Returns:
        An 8x3 JAX array (the conormally co-oriented restriction). If
        return_margin is True, returns (restriction, margin) with margin a
        scalar in [0, 1].
    """

    # --- Step 1: Compute determinants for all 56 combinations ---
    # Define a helper function to get the determinant for a single combination.
    # This function will be mapped over `ALL_COMBINATIONS`.
    def get_determinant_for_combination(combination_indices, matrix):
        # Select the specified columns from the input_matrix to form a 5x5 submatrix.
        submatrix = matrix[:, combination_indices]
        # Compute the determinant of the 5x5 submatrix.
        return jnp.linalg.det(submatrix)

    # Use `jax.vmap` to efficiently apply `get_determinant_for_combination`
    # across all 56 pre-computed combinations.
    # `in_axes=(0, None)` means:
    #   - The first argument (`combination_indices`) is mapped along its 0th axis (56 combinations).
    #   - The second argument (`input_matrix`) is broadcasted (used as-is for all mappings).
    all_determinants = jax.vmap(get_determinant_for_combination, in_axes=(0, None))(
        ALL_COMBINATIONS, eqlist_jacobian)
    # When the determinant is 0 there is a change that jnp.linalg.det will return
    # a nan instead. Convert these cases to 0 first.
    all_determinants = jnp.nan_to_num(all_determinants, nan=0.0)

    # --- Step 2: Find the combination with the largest |determinant| ---
    # Get the index of the maximum |determinant|.
    largest_det_idx = jnp.argmax(jnp.abs(all_determinants))
    # Retrieve the actual column indices for the combination with the largest |determinant|.
    ignored_coords = ALL_COMBINATIONS[largest_det_idx] # This is a 1D array of 5 indices

    # --- Step 3: Create an 8x8 identity matrix ---
    # Ensure the new matrix has the same data type as the input matrix.
    new_matrix = jnp.eye(8, dtype=eqlist_jacobian.dtype)

    # --- Step 4: Replace rows of the identity matrix ---
    # Replace the rows specified by `ignored_coords` with the 5 rows of the `input_matrix`.
    # JAX's `.at[indices].set(values)` is the idiomatic and efficient way to perform
    # scatter-like updates on JAX arrays.
    new_matrix = new_matrix.at[ignored_coords].set(eqlist_jacobian)

    # --- Step 5: Compute the inverse of the new 8x8 matrix ---
    # `jnp.linalg.inv` computes the inverse.
    inverted_matrix = jnp.linalg.inv(new_matrix)

    # --- Step 6: Delete columns specified by ignored_coords ---
    # First, determine which columns to keep.
    all_column_indices_8 = jnp.arange(8, dtype=jnp.int32)
    # Create a boolean mask: True for columns to keep, False for columns to delete.
    # `jnp.isin` checks if elements of `all_column_indices_8` are present in `ignored_coords`.
    # We want the *complement* (~) of those that are in `ignored_coords`.
    kept_cols_mask = ~jnp.isin(all_column_indices_8, ignored_coords)
    # Get the actual indices of the columns to keep (these will be 3 indices).
    # `jnp.where` returns a tuple, so we take the first element `[0]`.
    kept_cols = jnp.where(kept_cols_mask, size=3, fill_value=0)[0]
    # Select only the `kept_cols` from the `inverted_matrix`.
    # This results in an 8x3 JAX array.
    restriction = inverted_matrix[:, kept_cols]

    # --- Step 7: Conormal co-orientation (pin the +-1 orientation gauge) ---
    # restriction's columns span ker(J) = T_pL, but the argmax-|det| column
    # pick above leaves their *orientation* arbitrary -- it flips as the pick
    # changes from point to point along L, scattering the Omega|_L phase between
    # theta and theta+pi. Fix it geometrically with the equation-adapted normal
    # frame: orient T_pL so that
    #     det[ J^T | restriction ] > 0,
    # i.e. (eq-gradients, tangent-basis) is positively oriented in R^8 = C^4
    # (the canonical complex orientation). This is the conormal co-orientation
    # of L as the zero set of the 5 real equations. Unlike a fixed coordinate
    # 3-form, J^T spans (ker J)^perp, so the stacked 8x8 is nonsingular *exactly*
    # where rank(J) = 5 -- wherever L is transverse; it can only flip on the
    # non-transverse locus (rank J < 5), where L degenerates as a submanifold
    # (and where this routine's 8x8 inverse is already ill-posed).
    #
    # Negating one column is a diag(-1,1,1) congruence on the tangent basis, so
    # it is bit-identical for magnitude consumers (||R^T Omega R||_F and
    # sqrt(det(R^T G R))); the *only* observable effect is the now-consistent
    # sign of det(d w_c / d t) downstream, i.e. the Omega|_L phase mod 2pi.
    # Columns are unit-normalised before the det so only the sign and a
    # scale-free [0,1] margin are read off (Hadamard: |det| <= 1, ->0 at
    # degeneracy).
    conormal = jnp.concatenate([eqlist_jacobian.T, restriction], axis=1)  # (8, 8)
    col_norms = jnp.linalg.norm(conormal, axis=0, keepdims=True)
    conormal_unit = conormal / jnp.where(col_norms == 0.0, 1.0, col_norms)
    coorient_det = jnp.linalg.det(conormal_unit)
    flip = jnp.where(coorient_det < 0.0, -1.0, 1.0)
    restriction = restriction.at[:, 0].multiply(flip)

    if return_margin:
        return restriction, jnp.abs(coorient_det)
    return restriction

def compute_Omega_restriction(restriction: jnp.ndarray, Omega_coord: jnp.ndarray):
    """
    Computes the restriction applied to the holomorphic 3-form from the full restriction.

    Division of labour for the Omega|_L phase sign:
      - the deterministic (-1)^(patch_idx + max_idx) factor lives in
        compute_holomorphic_form;
      - the +-1 orientation gauge of T_pL is pinned in compute_restriction,
        which returns a *conormally co-oriented* tangent basis (oriented by the
        equation normals against the complex orientation of C^4);
      - this function then contributes det(dw_c/dt) on that basis, whose phase
        is therefore geometrically well defined wherever L is transverse.

    The conormal co-orientation supersedes the earlier coordinate-reference
    attempts (a fixed C=(0,1,2), or a per-patch argmax-min 3-tuple), which
    oriented against fixed coordinate 3-planes and so flipped on a codim-1
    coordinate-artifact locus of generic L. The conormal sign can only flip
    where rank(J) < 5 (L non-transverse) -- the genuine degeneracy locus,
    diagnostic rather than artifact. The mod-pi consumer
    (compute_special_condition_fitness) is invariant to this +-1 either way.

    Args:
        restriction: (N, 8, 3) array of restriction matrices.
        Omega_coord: (N, 3) array of the 3 affine-complex-coord indices for
                     the holomorphic-3-form wedge.

    Return:
        An (N,) complex array: phase carrier of Omega restricted to L
        (up to the +-1 orientation gauge).
    """
    Omega_coord_y = Omega_coord + 4
    N = restriction.shape[0]
    row = jnp.arange(N)[:, None]
    jacobian = restriction[row, Omega_coord] + 1j * restriction[row, Omega_coord_y]
    # Rescale the jacobian in case the determinant blows up
    max_abs_vals = jnp.max(jnp.abs(jacobian), axis=(-2, -1))
    safe_max_vals = jnp.where(max_abs_vals == 0, 1.0, max_abs_vals)
    scaling_factors = safe_max_vals[:, None, None]
    jacobian_scaled = jacobian / scaling_factors
    return jnp.linalg.det(jacobian_scaled)


