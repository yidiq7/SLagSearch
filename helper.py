import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np


# ----------------------------------------------------------------------------
# Point-cloud data location.
#
# POINTS_DIR is the one directory every Dwork-family helper resolves against.
# Each consumer script has a single POINTS_FILE variable at the top — edit
# that line (or override --points_file on the CLI) to point at any pickle.
#
# dwork_points_path / dwork_filename are scoped to the one-parameter Dwork
# pencil of the quintic (Σ z_i^5 + ψ · Π z_i = 0). For CICY / other families,
# either write a parallel constructor or just assign
#     POINTS_FILE = "your_data.pkl"
# directly in the consumer script — there is nothing magic about going
# through dwork_points_path.
# ----------------------------------------------------------------------------

POINTS_DIR = "."  # default: current working directory


def dwork_filename(psi, seed: int) -> str:
    """Filename for a Dwork-family point cloud at (psi, seed).

    Integer-real psi keeps the legacy 'psi0', 'psi10' form. Fractional real
    and complex psi extend without colliding: 'psi0.5', 'psi1+2j', 'psi-1.5j'.
    """
    psi = complex(psi)
    if psi.imag == 0:
        if psi.real == int(psi.real):
            psi_str = f"{int(psi.real)}"
        else:
            psi_str = f"{psi.real:g}"
    else:
        psi_str = f"{psi.real:g}{psi.imag:+g}j"
    return f"1mil_patch_all_psi{psi_str}_seed{seed}.pkl"


def dwork_points_path(psi, seed: int = 1024, points_dir: str = None) -> str:
    """Path to a Dwork-family point cloud. Joins POINTS_DIR with dwork_filename."""
    if points_dir is None:
        points_dir = POINTS_DIR
    return os.path.join(points_dir, dwork_filename(psi, seed))


def load_points(path: str) -> jnp.ndarray:
    """Load a point-cloud pkl and return (N, 10) real coords on the default device.

    The on-disk format is the (N, 5) complex array produced by
    points_gen/points_generation.py. This helper converts to the (N, 10) real
    representation [Re | Im] that the rest of the pipeline consumes.
    """
    with open(path, "rb") as f:
        arr = np.asarray(pickle.load(f))
    arr = np.concatenate([np.real(arr), np.imag(arr)], axis=1)
    return jax.device_put(jnp.asarray(arr))


def load_ga_checkpoint(path: str) -> dict:
    """Load a GA checkpoint pkl without importing GA.py.

    GA checkpoints reference ``GA.Species`` via pickle's class registry, which
    normally requires ``GA.py`` to be importable. To avoid pulling in JAX +
    GA module-level constants just to inspect a checkpoint, we stub Species
    as a plain attribute container. The returned dict has the same keys as
    saved (``population``, ``generation``, ``key``, ``species_list``,
    ``speciation_threshold``); the species objects expose their attributes
    (``id``, ``representative``, ``best_fitness``, ``sigma``, ...) directly.
    """

    class _SpeciesStub:
        pass

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "Species":
                return _SpeciesStub
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return _Unpickler(f).load()


def top_species(species_list, top_k: int = None):
    """Return species sorted by ``best_fitness`` descending, optionally sliced to top_k.

    Works with both live ``GA.Species`` objects and the stub objects returned
    by ``load_ga_checkpoint``. ``top_k=None`` returns the full sorted list.
    """
    def fit(s):
        return float(getattr(s, "best_fitness", float("-inf")))

    sorted_list = sorted(species_list, key=fit, reverse=True)
    return sorted_list if top_k is None else sorted_list[:top_k]


def assert_metric_psi_compatible(metric: str, psi) -> None:
    """Reject metric/psi combinations that would silently give wrong results.

    The Donaldson k=4 balanced coefficients in `calculate_complex_metric_k4`
    are precomputed for the Fermat quintic (psi=0). Using them at psi != 0
    would yield a non-Ricci-flat metric without any obvious failure mode.
    """
    if metric == 'k4_fermat' and complex(psi) != 0:
        raise ValueError(
            f"metric='k4_fermat' is only valid for the Fermat quintic (psi=0), "
            f"but got psi={psi}. Use metric='FS' for the deformed quintic, or "
            f"extend calculate_complex_metric_k4 with psi-dependent coefficients."
        )


@jax.jit
def canonicalize_coeffs(A: jnp.ndarray) -> jnp.ndarray:
    """
    # Optimized RREF using vectorized operations.
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
    """
    return A

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


def generate_basis_second_order_single_point(point: jnp.ndarray) -> jnp.ndarray:
    """
    Generate second-order basis functions from points on Fermat quintic.
    These are polynomials of degree (2, 2) in the coordinates: z_i z_j z_m_bar z_n_bar.

    Args:
        point: (5,) complex array of points on the quintic

    Returns:
        basis: (225,) real array of basis functions
               - 105 Imaginary parts of (v_A * v_B_bar) for A < B
               - 120 Real parts of (v_A * v_B_bar) for A <= B
               Where v is the vector of 15 quadratic monomials.
    """
    # 1. Construct the vector of quadratic monomials (z_i * z_j)
    # ---------------------------------------------------------
    zi = point[:, None]
    zj = point[None, :]
    
    # Shape: (5, 5) representing all z_i * z_j
    quad_products = zi * zj
    
    # We only need the unique monomials. Since z_i*z_j = z_j*z_i, 
    # we take the upper triangle (including diagonal).
    # There are 5*(5+1)/2 = 15 such monomials.
    triu_indices_1 = jnp.triu_indices(5, k=0)
    
    # Shape: (15,)
    # Let's call this vector v. v_A corresponds to a pair (i,j)
    v = quad_products[triu_indices_1[0], triu_indices_1[1]]

    # 2. Construct the matrix of quartic products (v_A * v_B_bar)
    # ---------------------------------------------------------
    # This represents (z_i z_j) * (z_m_bar z_n_bar)
    v_A = v[:, None]       # (15, 1)
    v_B_bar = jnp.conj(v[None, :]) # (1, 15)
    
    # Shape: (15, 15)
    # This matrix M is Hermitian by construction.
    M = v_A * v_B_bar

    # 3. Extract Independent Real Basis Functions
    # ---------------------------------------------------------
    # Just like the first order case, we extract the independent real parameters
    # from this Hermitian matrix.
    
    # Imaginary parts: Strict upper triangle of the 15x15 matrix (k=1)
    # Count: 15 * 14 / 2 = 105
    triu_indices_imag = jnp.triu_indices(15, k=1)
    imag_basis = jnp.imag(M[triu_indices_imag[0], triu_indices_imag[1]])

    # Real parts: Upper triangle + diagonal of the 15x15 matrix (k=0)
    # Count: 15 * 16 / 2 = 120
    triu_indices_real = jnp.triu_indices(15, k=0)
    real_basis = jnp.real(M[triu_indices_real[0], triu_indices_real[1]])

    # Total size: 105 + 120 = 225
    return jnp.concatenate([imag_basis, real_basis])


# All (i, j, k) triples with i <= j <= k from {0,...,4}. C(5+2, 3) = 35.
_CUBIC_TRIPLES = jnp.array(
    [(i, j, k) for i in range(5) for j in range(i, 5) for k in range(j, 5)],
    dtype=jnp.int32,
)


def generate_basis_third_order_single_point(point: jnp.ndarray) -> jnp.ndarray:
    """
    Generate third-order basis functions from points on Fermat quintic.
    These are polynomials of degree (3, 3): z_i z_j z_k z_l_bar z_m_bar z_n_bar.

    Args:
        point: (5,) complex array of points on the quintic

    Returns:
        basis: (1225,) real array of basis functions
               - 595 Imaginary parts of (w_A * w_B_bar) for A < B
               - 630 Real parts of (w_A * w_B_bar) for A <= B
               Where w is the vector of 35 cubic monomials z_i z_j z_k (i<=j<=k).
    """
    # 1. Construct the vector of cubic monomials w_A = z_i * z_j * z_k.
    w = (point[_CUBIC_TRIPLES[:, 0]]
         * point[_CUBIC_TRIPLES[:, 1]]
         * point[_CUBIC_TRIPLES[:, 2]])  # (35,)

    # 2. Construct the (35, 35) Hermitian matrix of (w_A * w_B_bar).
    M = w[:, None] * jnp.conj(w[None, :])

    # 3. Extract independent real basis functions.
    # Imag parts: strict upper triangle (k=1). Count: 35 * 34 / 2 = 595.
    triu_indices_imag = jnp.triu_indices(35, k=1)
    imag_basis = jnp.imag(M[triu_indices_imag[0], triu_indices_imag[1]])

    # Real parts: upper triangle + diagonal (k=0). Count: 35 * 36 / 2 = 630.
    triu_indices_real = jnp.triu_indices(35, k=0)
    real_basis = jnp.real(M[triu_indices_real[0], triu_indices_real[1]])

    # Total size: 595 + 630 = 1225.
    return jnp.concatenate([imag_basis, real_basis])


# All (i, j, k, l) quadruples with i <= j <= k <= l from {0,...,4}. C(5+3, 4) = 70.
_QUARTIC_QUADRUPLES = jnp.array(
    [(i, j, k, l)
     for i in range(5)
     for j in range(i, 5)
     for k in range(j, 5)
     for l in range(k, 5)],
    dtype=jnp.int32,
)


def generate_basis_fourth_order_single_point(point: jnp.ndarray) -> jnp.ndarray:
    """
    Generate fourth-order basis functions from points on Fermat quintic.
    These are polynomials of degree (4, 4): z_i z_j z_k z_l * conj of same.

    Args:
        point: (5,) complex array of points on the quintic

    Returns:
        basis: (4900,) real array of basis functions
               - 2415 Imaginary parts of (u_A * u_B_bar) for A < B
               - 2485 Real parts of (u_A * u_B_bar) for A <= B
               Where u is the vector of 70 quartic monomials z_i z_j z_k z_l
               (i <= j <= k <= l).
    """
    # 1. Construct the vector of quartic monomials u_A = z_i * z_j * z_k * z_l.
    u = (point[_QUARTIC_QUADRUPLES[:, 0]]
         * point[_QUARTIC_QUADRUPLES[:, 1]]
         * point[_QUARTIC_QUADRUPLES[:, 2]]
         * point[_QUARTIC_QUADRUPLES[:, 3]])  # (70,)

    # 2. Construct the (70, 70) Hermitian matrix of (u_A * u_B_bar).
    M = u[:, None] * jnp.conj(u[None, :])

    # 3. Extract independent real basis functions.
    # Imag parts: strict upper triangle (k=1). Count: 70 * 69 / 2 = 2415.
    triu_indices_imag = jnp.triu_indices(70, k=1)
    imag_basis = jnp.imag(M[triu_indices_imag[0], triu_indices_imag[1]])

    # Real parts: upper triangle + diagonal (k=0). Count: 70 * 71 / 2 = 2485.
    triu_indices_real = jnp.triu_indices(70, k=0)
    real_basis = jnp.real(M[triu_indices_real[0], triu_indices_real[1]])

    # Total size: 2415 + 2485 = 4900.
    return jnp.concatenate([imag_basis, real_basis])


# Genotype widths per max-degree. Switch is by coeffs.shape[1] (static under jit).
_D1_END = 25
_D2_END = 250    # 25 + 225
_D3_END = 1475   # 250 + 1225
_D4_END = 6375   # 1475 + 4900


def evaluate_equations_single_point(point: jnp.ndarray, coeffs: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the five equations. The input points are real.

    The coefficient matrix may be (3, 25), (3, 250), (3, 1475), or (3, 6375),
    selecting d=1, d=1+2, d=1+2+3, or d=1+2+3+4 ansatz respectively. Dispatch
    is by static shape.
    """
    point_complex = point[:5] + 1j * point[5:]
    norm_sq = jnp.vdot(point_complex, point_complex).real
    n_coeffs = coeffs.shape[1]

    basis_d1 = generate_basis_single_point(point_complex) / norm_sq
    eqs_vec = coeffs[:, :_D1_END] @ basis_d1

    if n_coeffs >= _D2_END:
        basis_d2 = generate_basis_second_order_single_point(point_complex) / (norm_sq ** 2)
        eqs_vec = eqs_vec + coeffs[:, _D1_END:_D2_END] @ basis_d2

    if n_coeffs >= _D3_END:
        basis_d3 = generate_basis_third_order_single_point(point_complex) / (norm_sq ** 3)
        eqs_vec = eqs_vec + coeffs[:, _D2_END:_D3_END] @ basis_d3

    if n_coeffs >= _D4_END:
        basis_d4 = generate_basis_fourth_order_single_point(point_complex) / (norm_sq ** 4)
        eqs_vec = eqs_vec + coeffs[:, _D3_END:_D4_END] @ basis_d4

    cy = jnp.sum(point_complex**5) + psi * jnp.prod(point_complex)
    return jnp.concatenate([jnp.array([jnp.real(cy), jnp.imag(cy)]), eqs_vec])


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

@jax.jit
def calculate_distance(ind1, ind2):
    return jnp.linalg.norm(ind1.ravel() - ind2.ravel())


@jax.jit
def calculate_distance_matrix(pop1: jnp.ndarray, pop2: jnp.ndarray) -> jnp.ndarray:
    dist_to_reps = jax.vmap(calculate_distance, in_axes=(None, 0))
    dist_matrix = jax.vmap(dist_to_reps, in_axes=(0, None))(pop1, pop2) # all pop1 vs pop2
    return dist_matrix

