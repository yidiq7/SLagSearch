import sympy as sp
from typing import Optional, List
import jax
import jax.numpy as jnp
import itertools

def get_restriction_symbolic(constant_coord: int = 0, ignored_coord: Optional[List[int]] = None):

    # --- Validation and Default Value ---
    # The index is after taking out the constant coordinate
    if ignored_coord is None:
        ignored_coord = [3, 4, 5, 6, 7]

    # --- Runtime Validation Logic ---
    if not isinstance(ignored_coord, list):
        raise TypeError("ignored_coord must be a list of integers.")
    if len(ignored_coord) != 5:
        raise ValueError(f"ignored_coord must contain exactly 5 elements, but got {len(ignored_coord)}.")
    if len(set(ignored_coord)) != 5:
        raise ValueError(f"All elements in ignored_coord must be unique. Found duplicates in {ignored_coord}.")
    if not all(isinstance(i, int) and 0 <= i <= 7 for i in ignored_coord):
        raise ValueError("All elements in ignored_coord must be integers between 0 and 7.")

    z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
    Z = [z0,z1,z2,z3,z4]
    f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5

    X = sp.symbols('x0:5', real=True)
    Y = sp.symbols('y0:5', real=True)

    # The order of symbols in XY is [x0,x1,x2,x3,x4, y0,y1,y2,y3,y4]
    XY = X + Y
    # This is the 4 complex affine coordinates after choosing a patch
    # If we set z0 = 1, then the four affine coordinates will be
    # x1 to x4 and y1 to y4
    XY_affine = (X[:constant_coord] + X[constant_coord+1:] +
                 Y[:constant_coord] + Y[constant_coord+1:])

    # -----------------------------------------------------------
    # Gonstruct the five constrains in real coordinates

    substitutions = {z: x + sp.I*y for z, x, y in zip(Z, X, Y)}
    # Perform the substitution
    f_substituted = f.subs(substitutions)
    cy_real, cy_imag = f_substituted.expand(complex=True).as_real_imag()

    imag_basis_z = [sp.im(Z[i] * sp.conjugate(Z[j])) for i in range(5) for j in range(i + 1, 5)]
    real_basis_z = [sp.re(Z[i] * sp.conjugate(Z[j])) for i in range(5) for j in range(i, 5)]

    # The complete basis in terms of z
    basis_z = imag_basis_z + real_basis_z

    # The .doit() method evaluates the Re() and Im() functions
    basis_xy = [b.subs(substitutions).doit() for b in basis_z]

    A = sp.symbols('a0:25', real=True)
    B = sp.symbols('b0:25', real=True)
    C = sp.symbols('c0:25', real=True)

    # Create the full linear combination of the basis functions
    # linear_combination = sum(wk * bk for wk, bk in zip(w, basis_xy))
    eq1 = sp.Add(*[ak * basisk for ak, basisk in zip(A, basis_xy)])
    eq2 = sp.Add(*[bk * basisk for bk, basisk in zip(B, basis_xy)])
    eq3 = sp.Add(*[ck * basisk for ck, basisk in zip(C, basis_xy)])

    eq_list = [cy_real, cy_imag, eq1, eq2, eq3]

    # Since we have more than one ignored_coordinate, sympy subs()
    # cannot replace two coordinates simultaneously. As a result, if the first
    # expression contains the coordinate to be replaced by the second expression
    # that coordinate will also be replaced. So here we will create a temporary 
    # coordinate list W to avoid this issue

    V = sp.symbols('v0:8', real=True)
    ignored_coordinate_v = [V[i] for i in ignored_coord]

    local_coordinates_v = sp.Matrix(V).subs({coord: func for coord, func in zip(ignored_coordinate_v, eq_list)})
    # Replace the correct affine coordinates back
    local_coordinates = local_coordinates_v.subs({v: xy for v, xy in zip(V, XY_affine)})
    jacobian_matrix = local_coordinates.jacobian(XY_affine)
    restriction = jacobian_matrix._rep.inv().to_Matrix()
    for coord in reversed(ignored_coord):
        restriction.col_del(coord)
    # Expand the dimension from 8 to 10 to include the two real constant coordinates
    # (By default, Re(z0) and Im(z0)
    #embedding_matrix = sp.eye(10)
    #embedding_matrix.col_del(constant_coord+5)
    #embedding_matrix.col_del(constant_coord)
    #restriction = embedding_matrix * restriction

    #jac_ignored = jacobian_matrix.extract(ignored_coord, ignored_coord)
    #det_jac_ignored = jac_ignored.applyfunc(sp.factor).det(method='berkowitz')  
    #restriction_scaled = restriction * det_jac_ignored
    return restriction
    #return restriction_scaled, det_jac_ignored


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

def get_jacobian_symbolic(constant_coord: int = 0):

    z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
    Z = [z0,z1,z2,z3,z4]
    f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5

    X = sp.symbols('x0:5', real=True)
    Y = sp.symbols('y0:5', real=True)

    # The order of symbols in XY is [x0,x1,x2,x3,x4, y0,y1,y2,y3,y4]
    XY = X + Y
    # This is the 4 complex affine coordinates after choosing a patch
    # If we set z0 = 1, then the four affine coordinates will be
    # x1 to x4 and y1 to y4
    XY_affine = (X[:constant_coord] + X[constant_coord+1:] +
                 Y[:constant_coord] + Y[constant_coord+1:])

    # -----------------------------------------------------------
    # Gonstruct the five constrains in real coordinates

    substitutions = {z: x + sp.I*y for z, x, y in zip(Z, X, Y)}
    # Perform the substitution
    f_substituted = f.subs(substitutions)
    cy_real, cy_imag = f_substituted.expand(complex=True).as_real_imag()

    imag_basis_z = [sp.im(Z[i] * sp.conjugate(Z[j])) for i in range(5) for j in range(i + 1, 5)]
    real_basis_z = [sp.re(Z[i] * sp.conjugate(Z[j])) for i in range(5) for j in range(i, 5)]

    # The complete basis in terms of z
    basis_z = imag_basis_z + real_basis_z

    # The .doit() method evaluates the Re() and Im() functions
    basis_xy = [b.subs(substitutions).doit() for b in basis_z]

    A = sp.symbols('a0:25', real=True)
    B = sp.symbols('b0:25', real=True)
    C = sp.symbols('c0:25', real=True)

    # Create the full linear combination of the basis functions
    # linear_combination = sum(wk * bk for wk, bk in zip(w, basis_xy))
    eq1 = sp.Add(*[ak * basisk for ak, basisk in zip(A, basis_xy)])
    eq2 = sp.Add(*[bk * basisk for bk, basisk in zip(B, basis_xy)])
    eq3 = sp.Add(*[ck * basisk for ck, basisk in zip(C, basis_xy)])

    eq_list = [cy_real, cy_imag, eq1, eq2, eq3]

    eq_jacobian = sp.Matrix(eq_list).jacobian(XY_affine)
    return eq_jacobian
 

# Pre-compute all 56 combinations of 5 column indices from 0 to 7.
# This array is static and will not be recomputed during JAX transformations (like vmap).
all_column_indices_range = jnp.arange(8)
combinations_list = list(itertools.combinations(all_column_indices_range.tolist(), 5))
ALL_COMBINATIONS = jnp.array(combinations_list, dtype=jnp.int32) # Shape (56, 5)

def get_restriction(eqlist_jacobian: jnp.ndarray) -> jnp.ndarray:
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
