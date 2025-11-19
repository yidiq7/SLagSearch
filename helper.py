import jax
import jax.numpy as jnp

@jax.jit
def canonicalize_coeffs(A: jnp.ndarray) -> jnp.ndarray:
    """
    Optimized RREF using vectorized operations.
    """
    #A = coeffs.astype(jnp.float32)
    eps = 1e-10
    
    # Process columns one by one, tracking pivot row
    def process_column(carry, col):
        A, pivot_row = carry
        
        # Skip if no more rows
        skip = pivot_row >= 3
        
        def find_and_apply_pivot(args):
            A, pivot_row, col = args
            
            # Get column and find pivot from pivot_row down
            col_data = A[:, col]
            
            # Create mask for valid rows
            mask = jnp.arange(3) >= pivot_row
            abs_vals = jnp.where(mask, jnp.abs(col_data), -1.0)
            
            # Find best pivot
            best_idx = jnp.argmax(abs_vals)
            best_val = col_data[best_idx]
            
            # Check if valid pivot
            has_pivot = jnp.abs(best_val) > eps
            
            def do_elimination(A):
                # Swap rows using permutation
                indices = jnp.arange(3)
                indices = indices.at[pivot_row].set(best_idx)
                indices = indices.at[best_idx].set(pivot_row)
                A = A[indices]
                
                # Scale pivot row
                scale = A[pivot_row, col]
                A = A.at[pivot_row].set(A[pivot_row] / scale)
                
                # Vectorized elimination
                factors = A[:, col]
                mask = jnp.arange(3) != pivot_row
                elimination = jnp.outer(factors * mask, A[pivot_row])
                A = A - elimination
                
                return A, pivot_row + 1
            
            return jax.lax.cond(has_pivot, do_elimination, lambda x: (x, pivot_row), A)
        
        A_new, pivot_row_new = jax.lax.cond(
            ~skip,
            find_and_apply_pivot,
            lambda args: (args[0], args[1]),
            (A, pivot_row, col)
        )
        
        return (A_new, pivot_row_new), None
    
    # Scan through all columns
    (result, _), _ = jax.lax.scan(process_column, (A, 0), jnp.arange(25))
    
    return result


@jax.jit
def canonicalize_coeffs_QR_decomposition(coeffs: jnp.ndarray) -> jnp.ndarray:
    """
    Currently not in used
    Canonicalizes a 3x25 coefficient matrix by extracting a unique orthonormal basis
    for its row space and fixing sign and row order ambiguity in a JIT-compatible way.
    """
    # 1. Get orthonormal basis of the row space.
    q, _ = jnp.linalg.qr(coeffs.T, mode='reduced')
    canonical_q = q.T  # Shape: (3, 25)

    # 2. Fix sign ambiguity robustly using JAX-native operations.
    # We ensure the first non-zero element of each row is positive.
    # `jnp.argmax` on a boolean array gives the index of the first True.
    first_nonzero_indices = jnp.argmax(canonical_q != 0, axis=1)
    
    # Get the signs of those first non-zero elements.
    signs = jnp.sign(jnp.take_along_axis(canonical_q, first_nonzero_indices[:, None], axis=1)).squeeze()
    
    # Handle rows that might be all zeros (sign would be 0). Default to 1 to do no change.
    signs = jnp.where(signs == 0, 1.0, signs)
    
    # Apply the sign correction.
    signed_q = canonical_q * signs[:, jnp.newaxis]

    # 3. Fix row order ambiguity using lexicographical sort.
    # `jnp.lexsort` sorts by keys in reverse order. To sort by columns 0, 1, 2...
    # we provide the columns of the transposed matrix in reverse order.
    sorted_indices = jnp.lexsort(signed_q.T[::-1, :])
    
    # Apply the sort to the rows to get the final canonical form.
    fully_canonical_q = signed_q[sorted_indices]

    return fully_canonical_q


def reconstruct_hermitian_matrices(coeffs_array: jnp.array) -> jnp.array:
    """
    Reconstructs a batch of 5x5 complex Hermitian matrices from coefficients.

    This function takes an array of shape (N, 25) and returns an array
    of shape (N, 5, 5), where N is the number of matrices.

    Args:
        coeffs_array: A jnp.array of shape (N, 25). Each row contains the
                      25 coefficients for one Hermitian matrix.

    Returns:
        A jnp.array of shape (N, 5, 5) containing the batch of reconstructed
        Hermitian matrices.
    """
    # Ensure the input has the correct dimensions (N, 25)
    if coeffs_array.ndim != 2 or coeffs_array.shape[1] != 25:
        raise ValueError("Input must be a 2D array with shape (N, 25).")

    # Define the reconstruction logic for a *single* matrix (one row).
    # This is often done as a private helper function or a local function.
    def _reconstruct_single(coeffs: jnp.array) -> jnp.array:
        
        # Start with an empty 5x5 complex matrix
        H = jnp.zeros((5, 5), dtype=jnp.complex64)
        
        # Split coefficients into imaginary and real parts
        imag_coeffs = coeffs[:10]
        real_coeffs = coeffs[10:]

        # 1. Populate imaginary parts for the strictly upper triangle (i < j)
        imag_idx = 0
        for i in range(5):
            for j in range(i + 1, 5):
                H = H.at[i, j].add(-1j * imag_coeffs[imag_idx] / 2)
                imag_idx += 1

        # 2. Populate real parts for the diagonal and upper triangle (i <= j)
        real_idx = 0
        for i in range(5):
            for j in range(i, 5):
                if i == j:
                    H = H.at[i, j].add(real_coeffs[real_idx])
                else:
                    H = H.at[i, j].add(real_coeffs[real_idx] / 2)
                real_idx += 1
        
        # 3. Complete the matrix using the Hermitian property H = H_upper + H_upper^†
        H_upper = jnp.triu(H, k=1)
        H_final = H + jnp.conjugate(H_upper.T)

        return H_final

    # Use vmap to "vectorize" the single-reconstruction function, telling it
    # to map over the first axis (the rows) of the input `coeffs_array`.
    reconstruct_batch = jax.vmap(_reconstruct_single, in_axes=0)
    
    # Apply the vectorized function to the entire batch of coefficients
    return reconstruct_batch(coeffs_array)

def format_array_with_commas(arr):
    if not isinstance(arr, jnp.ndarray):
        return str(arr)
    
    if arr.ndim == 1:
        return f"[{', '.join(map(str, arr.tolist()))}]"
    
    formatted_rows = [format_array_with_commas(row) for row in arr]
    return f"[{',\n'.join(formatted_rows)}]"


def generate_basis_single_point(point: jnp.ndarray) -> jnp.ndarray:
    """
    Generate basis functions from points on Fermat quintic.

    Args:
        points: (5,) complex array of points on the quintic

    Returns:
        basis: (25,) real array of basis functions
               First 10 are Im(zi*zjbar) for i<j, next 15 are Re(zi*zjbar) for i<=j
    """

    # Create all pairwise products zi * zj_bar using broadcasting
    # Shape: (5, 5)
    z_sq = jnp.sum(point * point.conj()).real # (N,)
    zi = point[:, None]  # (5, 1)
    zj_bar = jnp.conj(point[None, :])  # (1, 5)
    products = zi * zj_bar / z_sq # (5, 5)

    # Extract upper triangular indices for imaginary parts (i < j)
    # This gives us 10 unique imaginary parts
    triu_indices_imag = jnp.triu_indices(5, k=1)
    imag_basis = jnp.imag(products[triu_indices_imag[0], triu_indices_imag[1]])  # (10,)

    # Extract upper triangular indices including diagonal for real parts (i <= j)
    # This gives us 15 unique real parts
    triu_indices_real = jnp.triu_indices(5, k=0)
    real_basis = jnp.real(products[ triu_indices_real[0], triu_indices_real[1]])  # (15,)

    # Concatenate to form complete basis
    return jnp.concatenate([imag_basis, real_basis])  # (25,)


def evaluate_equations_single_point(point: jnp.ndarray, coeffs: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the five equations. The input points are real."""
    # W1: 25 x 10, W2: 10 x 3
    W1 = coeffs[0: 250].reshape(25, 10)
    b1 = coeffs[250:260]
    W2 = coeffs[260: 290].reshape(10, 3)
    b2 = coeffs[290:293]

    point_complex = point[:5] + 1j * point[5:]
    basis = generate_basis_single_point(point_complex)
    layer1 = jnp.tanh(basis @ W1 + b1)
    eqs_vec = layer1 @ W2 + b2
    cy = jnp.sum(point_complex**5) + psi * jnp.prod(point_complex)
    eqs_evaluated = jnp.array([jnp.real(cy), jnp.imag(cy), *eqs_vec]) 
    return eqs_evaluated


@jax.jit
def convert_real_to_complex_batch(points_real: jnp.ndarray) -> jnp.ndarray:
    """
    Converts (N, 10) real representation to (N, 5) complex.
    
    Args:
        points_real: An (N, 10) real array where first 5 are real parts, last 5 are imaginary
        
    Returns:
        An (N, 5) complex array
    """
    return points_real[:, :5] + 1j * points_real[:, 5:]


@jax.jit
def convert_complex_to_real_batch(points_complex: jnp.ndarray) -> jnp.ndarray:
    """
    Converts (N, 5) complex representation to (N, 10) real.
    
    Args:
        points_complex: An (N, 5) complex array
        
    Returns:
        An (N, 10) real array where first 5 are real parts, last 5 are imaginary
    """
    return jnp.concatenate([jnp.real(points_complex), jnp.imag(points_complex)], axis=1)


@jax.jit
def determine_patches_batch(points_complex: jnp.ndarray) -> jnp.ndarray:
    """
    Determines the appropriate patch for a batch of points.
    
    Args:
        points_complex: An (N, 5) complex array
        
    Returns:
        patch_indices: An (N,) integer array with values in [0,4]
    """
    magnitudes = jnp.abs(points_complex)
    patch_indices = jnp.argmax(magnitudes, axis=1)
    return patch_indices


def delete_index(arr, index):
    """
    JAX-compatible version of deleting an element at a specific index.

    This function works with traced values in JIT-compiled code.

    Args:
        arr: JAX array to remove element from (shape [n])
        index: Scalar index of element to remove (can be a traced value)

    Returns:
        Array with element at `index` removed (shape [n-1])
    """
    n = arr.shape[0]
    
    # Create indices for the *output* array, e.g., [0, 1, 2, 3] if n=5
    out_indices = jnp.arange(n - 1)
    
    # Create the corresponding indices for the *input* array
    # If the output index 'i' is before the removed 'index', we take arr[i].
    # If 'i' is at or after the removed 'index', we take arr[i+1].
    # jnp.where is JIT-compatible with a traced 'index'
    in_indices = jnp.where(out_indices < index, 
                          out_indices, 
                          out_indices + 1)
    
    # Gather elements from 'arr' using the calculated indices.
    # This integer array indexing is JIT-compatible.
    return arr[in_indices]

def convert_real_to_complex_single(points_real: jnp.ndarray) -> jnp.ndarray:
    """
    Converts (N, 10) real representation to (N, 5) complex.
    
    Args:
        points_real: An (N, 10) real array where first 5 are real parts, last 5 are imaginary
        
    Returns:
        An (N, 5) complex array
    """
    return points_real[:5] + 1j * points_real[5:]


def convert_complex_to_real_single(points_complex: jnp.ndarray) -> jnp.ndarray:
    """
    Converts (N, 5) complex representation to (N, 10) real.
    
    Args:
        points_complex: An (N, 5) complex array
        
    Returns:
        An (N, 10) real array where first 5 are real parts, last 5 are imaginary
    """
    return jnp.concatenate([jnp.real(points_complex), jnp.imag(points_complex)])


def determine_patch_and_rescale_single(point_complex: jnp.ndarray) -> tuple[jnp.ndarray, int]:
    """
    Determines the appropriate patch for a single point and rescales it.
    
    The patch is chosen so that the coordinate with largest magnitude is set to 1.
    This ensures numerical stability and proper projective coordinates.
    
    Args:
        point_complex: A (5,) complex array of homogeneous coordinates
        
    Returns:
        rescaled_point: A (5,) complex array with largest coordinate normalized to 1
        patch_index: Integer in [0,4] indicating which coordinate was normalized
    """
    magnitudes = jnp.abs(point_complex)
    patch_index = jnp.argmax(magnitudes)
    
    # Rescale so that point_complex[patch_index] has magnitude 1
    # Preserve the phase of the largest coordinate
    scale_factor = point_complex[patch_index]
    rescaled_point = point_complex / scale_factor
    
    return rescaled_point, patch_index
