import sympy as sp
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


def compute_affine_jacobian(p_10d: jnp.ndarray, coeffs: jnp.ndarray, psi: jnp.ndarray, patch_index: int) -> jnp.ndarray:
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
    affine_jacobian = select_active_jacobian(patch_indices, full_jacobian)
    
    return affine_jacobian 

# Pre-compute all 56 combinations of 5 column indices from 0 to 7.
# This array is static and will not be recomputed during JAX transformations (like vmap).
all_column_indices_range = jnp.arange(8)
combinations_list = list(itertools.combinations(all_column_indices_range.tolist(), 5))
ALL_COMBINATIONS = jnp.array(combinations_list, dtype=jnp.int32) # Shape (56, 5)

def compute_restriction(eqlist_jacobian: jnp.ndarray) -> jnp.ndarray:
    """
    Processes a 5x8 JAX array according to the specified steps:
    1. Finds the combination of 5 columns that forms a 5x5 submatrix
       with the largest determinant.
    2. Creates an 8x8 identity matrix and replaces rows corresponding
       to the chosen column indices with the rows of the input matrix.
    3. Computes the inverse of this new 8x8 matrix.
    4. Deletes the columns from the inverted matrix that correspond
       to the chosen column indices, resulting in an 8x3 array.

    This function is designed to be JAX-transformable and efficient,
    especially when used with `jax.vmap`.

    Args:
        eqlist_jacobian: A 5x8 JAX array of floating-point numbers.

    Returns:
        An 8x3 JAX array, which is the result of the described operations.
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

    # --- Step 2: Find the combination with the largest determinant ---
    # Get the index of the maximum determinant.
    largest_det_idx = jnp.argmax(all_determinants)
    # Retrieve the actual column indices for the combination with the largest determinant.
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

    return restriction

def compute_Omega_restriction(restriction: jnp.ndarray, Omega_coord: jnp.ndarray):
    """
    Computes the restriction applied to the holomorphic 3-form from the full restriction

    Args:
        restriction: An (N, 8, 8) array of the numerical restriction matrices.
        Omega_coord: The index of the affine coordinates used to represent the holomorphic
                     3-form. For example, With z_4 being the dependent coordinate,
                     this list would be [0, 1, 2]. If z_3 is chosen as the dependent coordinate,
                     then the list would be [0, 1, 3]. It doesn't depend on patch
                     (which coordinate set to 1.)

    Return:
        An (N,) array of the determinant of the jacobian.
    """
    Omega_coord_y = Omega_coord + 4
    N = restriction.shape[0]
    row = jnp.arange(N)[:, None]
    jacobian = restriction[row, Omega_coord] + 1j*restriction[row, Omega_coord_y]
    # Rescale the jacobian in case the determinant blows up
    max_abs_vals = jnp.max(jnp.abs(jacobian), axis=(-2, -1))
    safe_max_vals = jnp.where(max_abs_vals == 0, 1.0, max_abs_vals)
    scaling_factors = safe_max_vals[:, None, None]
    jacobian_scaled = jacobian / scaling_factors
    return jnp.linalg.det(jacobian_scaled)


