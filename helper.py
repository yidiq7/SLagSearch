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
    zi = point[:, None]  # (5, 1)
    zj_bar = jnp.conj(point[None, :])  # (1, 5)
    products = zi * zj_bar  # (5, 5)

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
    point_complex = point[:5] + 1j * point[5:]
    basis = generate_basis_single_point(point_complex)
    eqs_vec = coeffs @ basis # (3,)
    cy = jnp.sum(point_complex**5) + psi * jnp.prod(point_complex)
    eqs_evaluated = jnp.array([jnp.real(cy), jnp.imag(cy), *eqs_vec]) 
    return eqs_evaluated

