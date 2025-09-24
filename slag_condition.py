import jax
import jax.numpy as jnp
from get_restriction import compute_jacobian, compute_restriction, compute_Omega_restriction
from functools import partial
import math
from itertools import combinations_with_replacement
from collections import Counter

# --- Core Calculation ---

def calculate_complex_metric_old(z: jnp.ndarray, patch_index: int) -> jnp.ndarray:
    """
    Calculates the complex components g_ab_bar of the Fubini-Study metric.

    This is the fundamental quantity from which both the metric tensor and the
    Kähler form can be derived. It follows the standard definition:
    g_{k,l_bar} = d_k d_l_bar log(1 + |zeta|^2)

    Args:
        z: A (5,) array of complex numbers (homogeneous coordinates).
        patch_index: The index for the affine patch.

    Returns:
        A (4, 4) complex array for the Hermitian metric g_ab_bar.
    """
    # The coordinate for the patch denominator
    z_patch = z[patch_index]

    # Inhomogeneous coordinates (zeta) are the other 4 coordinates divided by z_patch
    zeta = jnp.delete(z, patch_index)

    # Denominator for the metric formula: 1 + |zeta|^2
    norm_sq_zeta = 1.0 + jnp.sum(jnp.abs(zeta)**2)

    # The Fubini-Study metric in complex coordinates is given by:
    # g_ab_bar = ( (1+|zeta|^2) * delta_ab - conj(zeta_a) * zeta_b ) / (1+|zeta|^2)^2
    identity = jnp.eye(4, dtype=zeta.dtype)
    # The jnp.outer(conj(a), b) produces a matrix with entries M[i,j] = conj(a[i]) * b[j]
    outer_prod = jnp.outer(jnp.conj(zeta), zeta)
    g_complex = (identity * norm_sq_zeta - outer_prod) / (norm_sq_zeta**2)
    return g_complex

def calculate_complex_metric_FS(z: jnp.ndarray, patch_index: int) -> jnp.ndarray:
    """
    Calculates the complex components g_ab_bar of the Fubini-Study metric
    by differentiating the Kähler potential.

    This is the fundamental quantity from which both the metric tensor and the
    Kähler form can be derived. It follows the standard definition:
    g_{a,b_bar} = d_a d_b_bar K
    where K is the Kähler potential, K = log(1 + |zeta|^2).

    This implementation uses JAX's automatic differentiation to compute the
    second derivatives of K, rather than using the closed-form solution.

    Args:
        z: A (5,) array of complex numbers (homogeneous coordinates).
        patch_index: The index for the affine patch.

    Returns:
        A (4, 4) complex array for the Hermitian metric g_ab_bar.
    """
    # The coordinate for the patch denominator
    z_patch = z[patch_index]

    # Inhomogeneous coordinates (zeta) are the other 4 coordinates divided by z_patch
    zeta = jnp.delete(z, patch_index)

    def kahler_potential(zeta_coords: jnp.ndarray, zeta_bar_coords: jnp.ndarray) -> float:
        """
        Defines the Kähler potential for the Fubini-Study metric.
        K = log(1 + |zeta|^2) = log(1 + sum(zeta * conj(zeta)))
        """
        # We treat zeta and its conjugate as independent variables for differentiation.
        norm_sq = 1.0 + jnp.sum(zeta_coords * zeta_bar_coords)
        return jnp.log(norm_sq)

    metric_func = jax.jacfwd(jax.grad(kahler_potential, argnums=0, holomorphic=True), argnums=1, holomorphic=True)

    # Evaluate the metric function at the given coordinates
    g_complex = metric_func(zeta, jnp.conj(zeta))
    return g_complex

def generate_quintic_exponents(num_vars: int = 5, degree: int = 4) -> jnp.ndarray:
    """
    Generates the exponents for all unique monomials for a polynomial.

    For a quintic polynomial in 5 variables, this will produce a (126, 5) array,
    where each row is the set of exponents for one monomial term.
    (e.g., [5,0,0,0,0], [4,1,0,0,0], etc.)

    Args:
        num_vars: The number of variables in the polynomial (e.g., 5).
        degree: The degree of the polynomial (e.g., 5).

    Returns:
        A jax.numpy array of shape (N, num_vars) where N is the number of
        unique monomials.
    """
    exponents = []
    # Generates all combinations of variable indices for a given degree
    for combo in combinations_with_replacement(range(num_vars), degree):
        # Counts occurrences of each variable index to get the powers
        count = Counter(combo)
        exponent_tuple = tuple(count.get(i, 0) for i in range(num_vars))
        exponents.append(exponent_tuple)

    exponents.sort(key=max)
    return jnp.array(exponents, dtype=jnp.int32)

def _create_coefficient_mapping(exponents: jnp.ndarray):
    """
    Creates a mapping from the 126 monomials to the 7 unique coefficient types.
    """
    # Get the canonical form of each exponent by sorting it descending.
    # Ex: [1,2,1,1,0] -> (2,1,1,1,0)
    sorted_exponents = [tuple(sorted(exp.tolist(), reverse=True)) for exp in exponents]

    # Find the set of unique canonical forms and sort them to create a stable order.
    # This determines the order for the 7-element input coefficient vector.
    # The key sorts first by highest power, then lexicographically.
    canonical_forms = sorted(list(set(sorted_exponents)), key=lambda x: (x[0], x))

    # Create a map from the canonical form to its index (0-6)
    form_to_index = {form: i for i, form in enumerate(canonical_forms)}

    # For each of the 126 monomials, find the index of its canonical form.
    # This array will be used to "gather" coefficients.
    mapping_indices = jnp.array([form_to_index[se] for se in sorted_exponents])
    return jnp.array(canonical_forms, dtype=jnp.int32), mapping_indices

# --- Pre-compute constants at module load time for maximum performance ---

# (7, 5) array of the unique, sorted exponent structures.
# (126,) array of indices (0-6) mapping each monomial to its canonical form.
_QUINTIC_EXPONENTS = generate_quintic_exponents(num_vars=5, degree=4)
_CANONICAL_EXPONENT_FORMS, _COEFF_MAPPING_INDICES = _create_coefficient_mapping(_QUINTIC_EXPONENTS)


def calculate_complex_metric_k4(z: jnp.ndarray, patch_index: int) -> jnp.ndarray:
    """
    Calculates the complex metric g_ab_bar from a quintic polynomial Kähler potential
    with symmetrized coefficients.

    Args:
        z: A (5,) array of complex homogeneous coordinates.
        patch_index: The index for the affine patch.
        
    Returns:
        A (4, 4) complex array for the Hermitian metric g_ab_bar.
    """
    """
        unique_coeffs: A (7,) array of unique complex coefficients. The order
                   corresponds to the canonical exponent forms, which can be
                   inspected by printing `_CANONICAL_EXPONENT_FORMS`. The order is:
                   [0]: (1,1,1,1,1)
                   [1]: (2,1,1,1,0)
                   [2]: (2,2,1,0,0)
                   [3]: (3,1,1,0,0)
                   [4]: (3,2,0,0,0)
                   [5]: (4,1,0,0,0)
                   [6]: (5,0,0,0,0)
    """

    #unique_coeffs = jnp.array([-4.79909*240, -75.298664*12, -83.726102*8, -103.669506*8, -39.049639*12, -33.852379*12, 1.0*(-180)])
    unique_coeffs = jnp.array([2.214272*48, 14.010661*8, 2.3827940*24, 9.073280*12, 1.0*48])
    # Expand the 7 unique coefficients into the full 126-coefficient array.
    # This is a highly efficient gather operation in JAX.
    full_coeffs = unique_coeffs[_COEFF_MAPPING_INDICES]

    # Inhomogeneous coordinates (zeta)
    z_patch = z[patch_index]
    zeta = jnp.delete(z, patch_index) / z_patch

    def kahler_potential(zeta_coords: jnp.ndarray, zeta_bar_coords: jnp.ndarray) -> float:
        """
        Defines K = log(sum_i gamma_i |m_i|^2).
        This function is structured to be compatible with jax.grad on complex vars.
        """
        # Re-homogenize both the holomorphic and anti-holomorphic coordinates
        full_coords = jnp.insert(zeta_coords, patch_index, 1.0)
        full_coords_bar = jnp.insert(zeta_bar_coords, patch_index, 1.0)

        # Vectorize the monomial calculation over all 126 exponent sets.
        vmap_mono = jax.vmap(lambda exp, base: jnp.prod(base ** exp), in_axes=(0, None))
        
        # Calculate all 126 monomial values m_i(zeta)
        monomial_values = vmap_mono(_QUINTIC_EXPONENTS, full_coords)
        
        # Calculate all 126 anti-monomial values m_i(zeta_bar)
        monomial_values_bar = vmap_mono(_QUINTIC_EXPONENTS, full_coords_bar)
        
        # Calculate the real potential sum_i gamma_i * m_i * m_i_bar
        # jnp.real is used for type stability, although the product is already real.
        potential_sum = jnp.sum(full_coeffs * monomial_values * monomial_values_bar)
        
        return jnp.log(potential_sum)

    # The differentiation logic remains the same, computing d/d(zeta) d/d(zeta_bar) K
    metric_func = jax.jacfwd(jax.grad(kahler_potential, argnums=0, holomorphic=True), argnums=1, holomorphic=True)
    
    # Evaluate the metric function at the given coordinates
    g_complex = metric_func(zeta, jnp.conj(zeta))
    
    return g_complex

# --- Assembly Helpers ---

def _assemble_metric_tensor(g_complex: jnp.ndarray) -> jnp.ndarray:
    """Assembles the 8x8 real metric tensor G from the 4x4 complex metric."""
    g_real = jnp.real(g_complex)
    g_imag = jnp.imag(g_complex)
    # The real metric G is a block matrix: G = [[ R, -I ], [ I,  R ]]
    top_block = jnp.concatenate([g_real, g_imag], axis=1)
    bottom_block = jnp.concatenate([-g_imag, g_real], axis=1)
    return jnp.concatenate([top_block, bottom_block], axis=0)

def _assemble_kahler_form(g_complex: jnp.ndarray) -> jnp.ndarray:
    """Assembles the 8x8 real Kähler form Omega from the 4x4 complex metric."""
    g_real = jnp.real(g_complex)
    g_imag = jnp.imag(g_complex)
    # The Kähler form Omega = G*J, which results in: Omega = [[-I, -R], [R, -I]]
    top_block = jnp.concatenate([-g_imag, g_real], axis=1)
    bottom_block = jnp.concatenate([-g_real, -g_imag], axis=1)
    return jnp.concatenate([top_block, bottom_block], axis=0)

# --- Vmapped Single-Point Computers ---

def _compute_metric_for_single_point(z: jnp.ndarray, patch_index: int, metric: str, epsilon: float) -> jnp.ndarray:
    """Computes the Fubini-Study metric tensor G for a single point."""
    return jax.lax.cond(
        jnp.abs(z[patch_index]) < epsilon,
        lambda: jnp.full((8, 8), jnp.nan, dtype=jnp.float32),
        lambda: _assemble_metric_tensor(calculate_complex_metric_k4(z, patch_index))
    )

def _compute_kahler_for_single_point(z: jnp.ndarray, patch_index: int, metric: str, epsilon: float) -> jnp.ndarray:
    """Computes the Fubini-Study Kähler form Omega for a single point."""
    # Fubini-Study metric
    if metric == 'FS':
        return jax.lax.cond(
            jnp.abs(z[patch_index]) < epsilon,
            lambda: jnp.full((8, 8), jnp.nan, dtype=jnp.float32),
            lambda: _assemble_kahler_form(calculate_complex_metric_FS(z, patch_index))
        )
    # k = 4 Ricci-flat metric
    elif metric == 'k4':
        return jax.lax.cond(
            jnp.abs(z[patch_index]) < epsilon,
            lambda: jnp.full((8, 8), jnp.nan, dtype=jnp.float32),
            lambda: _assemble_kahler_form(calculate_complex_metric_k4(z, patch_index))
        )
    else:
       raise ValueError(f"Unsupported metric: '{metric}'. Options are 'FS' or 'k4'.") 


# --- Public API Functions ---

def compute_fubini_study_metric(
    points_Z: jnp.ndarray,
    patch_index: int = 0,
    epsilon: float = 1e-8
) -> jnp.ndarray:
    """
    Computes the Fubini-Study metric tensor G for points in CP^4.

    Args:
        points_Z: An (N, 5) array of complex numbers (homogeneous coordinates).
        patch_index: The index of the homogeneous coordinate to use for the affine patch.
        epsilon: A small number to avoid division by zero.

    Returns:
        An (N, 8, 8) array of real numbers. Each 8x8 matrix is the metric tensor `G`.
    """
    vmapped_computer = jax.vmap(
        _compute_metric_for_single_point, in_axes=(0, None, None)
    )
    return vmapped_computer(points_Z, patch_index, epsilon)


def compute_kahler_form(
    points_Z: jnp.ndarray,
    patch_index: int = 0,
    metric: str = 'FS',
    epsilon: float = 1e-8
) -> jnp.ndarray:
    """
    Computes the Fubini-Study Kähler form Omega for points in CP^4.

    The Kähler form is a 2-form, represented here by its component matrix Omega_ij.
    This matrix is always antisymmetric.

    Args:
        points_Z: An (N, 5) array of complex numbers (homogeneous coordinates).
        patch_index: The index of the homogeneous coordinate to use for the affine patch.
        epsilon: A small number to avoid division by zero.

    Returns:
        An (N, 8, 8) array of real numbers. Each 8x8 matrix is the component
        matrix `Omega` of the Kähler form.
    """
    vmapped_computer = jax.vmap(
        _compute_kahler_for_single_point, in_axes=(0, None, None, None)
    )
    return vmapped_computer(points_Z, patch_index, metric, epsilon)

def compute_kahler_form_restricted(points: jnp.ndarray, restriction: jnp.ndarray, constant_coord: int = 0, metric: str = 'FS') -> jnp.ndarray:
    jit_compute_kahler_form = jax.jit(compute_kahler_form, static_argnums=(1, 2))
    kahler_form = jit_compute_kahler_form(points, constant_coord, metric)
    kahler_form_restricted = jnp.einsum('nij,nik,njl->nkl', kahler_form, restriction, restriction)
    return kahler_form_restricted


def compute_kahler_form_unrestricted(points: jnp.ndarray, constant_coord: int = 0, metric: str = 'FS') -> jnp.ndarray:
    jit_compute_kahler_form = jax.jit(compute_kahler_form, static_argnums=(1, 2))
    kahler_form = jit_compute_kahler_form(points, constant_coord, metric)
    return kahler_form


def compute_lagrangian_condition_fitness(kahler_form_unrestricted: jnp.ndarray, restriction: jnp.ndarray, k: int=10):
    kahler_form_restricted = jnp.einsum('nij,nik,njl->nkl', kahler_form_unrestricted, restriction, restriction)
    frobenius_norms = jnp.linalg.norm(kahler_form_restricted, axis=(1, 2))
    normalization_factor = jnp.linalg.norm(kahler_form_unrestricted, axis=(1, 2))
    norms_normalized = frobenius_norms / normalization_factor  
    # Remove the potential blow-up
    sorted_norms = jnp.sort(norms_normalized)
    norms_cut = sorted_norms[:int(sorted_norms.shape[0]*0.99)]
    kahler_form_loss = jnp.mean(norms_cut)
    #kahler_form_loss = jnp.mean(norms_normalized)
    fitness = jnp.exp(-k*kahler_form_loss)
    return fitness


def get_Omega_coord(min_idx):
    # Choose the rest three coordinates to form the basis
    coord_lookup_table = jnp.array([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2]
    ])
    return coord_lookup_table[min_idx]


def compute_holomorphic_form(points_complex: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the holomorphic 3-form from given points 

    Args:
        points: An (N, 5) complex array 
    Return:
        An (N,) array containing the component of the holomorphic 3-from
        Omega = 1 / (5 * z_i)**4 dz_1 ^ ... dz_{i-1} ^ dz_{i+1} ... ^ dz_4
        where the index i is chosen so that the norm of Omega is the smallest

        An (N,) array containing the index i
        An (N, 3) array containing the three coordinates in the basis
    """

    Omega = 1 / (5*(points_complex[:, 1:])**4)
    Omega_min_indices = jnp.argmin(jnp.abs(Omega), axis=1)
    Omega = Omega[jnp.arange(Omega.shape[0]), Omega_min_indices]
    Omega_coord = jax.vmap(get_Omega_coord)(Omega_min_indices)

    return Omega, Omega_min_indices, Omega_coord


def compute_holomorphic_form_restricted(points_complex: jnp.ndarray, restriction: jnp.ndarray, phase_only: bool):

    Omega, Omega_min_indices, Omega_coord = compute_holomorphic_form(points_complex)
    Omega_restriction = compute_Omega_restriction(restriction, Omega_coord)
    if phase_only:
        phase_Omega = -4*jnp.angle(points_complex[jnp.arange(points_complex.shape[0]), Omega_min_indices+1])
        phase_restriction = jnp.angle(Omega_restriction)
        phase = phase_Omega + phase_restriction
        phase = phase % (2*jnp.pi)
        return phase
    else:
        print("Warning: There is a scaling factor in Omega_restriction, "
            "which does not affect the phase but the actual Omega might be wrong. "
            "Don't use this until further checking.")
        Omega_restricted = Omega * Omega_restriction
        return Omega_restricted

@partial(jax.jit, static_argnames=('n_bins',))
def compute_special_condition_fitness(phases: jnp.array, n_bins: int=100) -> jnp.float32:
    """
    Calculates fitness based on the concentration of angles.

    Args:
        angles: A jnp.array of angles in the range [0, pi].
        n_bins: The number of bins to use for the histogram.

    Returns:
        A scalar fitness value. High fitness means high concentration.
    """
    # Create a histogram to approximate the distribution
    phases = phases % jnp.pi
    counts, _ = jnp.histogram(phases, bins=n_bins, range=(0, jnp.pi))

    # Calculate the probability distribution
    probs = counts / jnp.sum(counts)

    # Calculate Shannon entropy
    # Add a small epsilon to avoid log(0) for empty bins
    epsilon = 1e-9
    entropy = -jnp.sum(probs * jnp.log(probs + epsilon))

    # Calculate the maximum possible entropy (for a uniform distribution)
    max_entropy = jnp.log(n_bins)

    #fitness = max_entropy - entropy
    fitness = 1 - entropy / max_entropy 
    # For RP^3 the max fitness is around 2.37.
    # With a perturbation of 0.001 the fitness is around 0.788

    return fitness

vmap_compute_jacobian = jax.vmap(compute_jacobian, in_axes=(0, None, None, None))
vmap_compute_restriction = jax.vmap(compute_restriction, in_axes=0)

def compute_combined_fitness(min_set_real: jnp.ndarray, coeffs: jnp.ndarray, psi: jnp.ndarray, constant_coord: int=0, metric: str='FS', debug_mode: bool=False) -> jnp.float32:
    jacobians = vmap_compute_jacobian(min_set_real, coeffs, psi, constant_coord)
    restriction = vmap_compute_restriction(jacobians)

    min_set = min_set_real[:, :5] + 1j*min_set_real[:, 5:]
    kahler_form_unrestricted = compute_kahler_form_unrestricted(min_set, constant_coord=constant_coord, metric=metric)
    lagrangian_fitness = compute_lagrangian_condition_fitness(kahler_form_unrestricted, restriction, k=10)

    phases = compute_holomorphic_form_restricted(min_set, restriction, phase_only=True)
    n_bins_val = 100
    special_fitness = compute_special_condition_fitness(phases, n_bins=n_bins_val)
   
    #combined_fitness = special_fitness
    combined_fitness = lagrangian_fitness * special_fitness 
    #combined_fitness = jnp.where(lagrangian_fitness > 0.98 , 1 + special_fitness, lagrangian_fitness)

    if debug_mode:
        kahler_form_restricted = compute_kahler_form_restricted(min_set, restriction, constant_coord=constant_coord, metric=metric)
        normalization_factor = jnp.linalg.norm(kahler_form_unrestricted, axis=(1, 2))
        kahler_form_restricted_normalized = kahler_form_restricted / jnp.sqrt(normalization_factor[:, None, None])
        # Test
        #kahler_form_restricted_normalized = kahler_form_restricted 
        #kahler_form_unrestricted_normalized = compute_kahler_form_unrestricted(min_set, constant_coord=constant_coord)
        return combined_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted_normalized, restriction, phases
    else:
        return combined_fitness


# --- Example Usage ---
if __name__ == '__main__':
    # JIT-compile our functions for performance
    jit_compute_metric = jax.jit(compute_fubini_study_metric, static_argnums=(1,))
    jit_compute_kahler = jax.jit(compute_kahler_form, static_argnums=(1,))

    # Example point
    point = jnp.array([[1.0 + 0.1j, 0.5 - 0.2j, 0.2, 0.1, 0.1j]], dtype=jnp.complex64)

    print("--- Metric Tensor G ---")
    metric_tensor = jit_compute_metric(point, 0)
    print("Metric G for the point (rounded):")
    print(jnp.round(metric_tensor[0], 3))
    # Verify symmetry: G should be equal to its transpose
    is_symmetric = jnp.allclose(metric_tensor[0], metric_tensor[0].T)
    print(f"Is G symmetric? {is_symmetric}")


    print("\n--- Kähler Form Omega ---")
    kahler_matrix = jit_compute_kahler(point, 0)
    print("Kähler form Omega for the point (rounded):")
    print(jnp.round(kahler_matrix[0], 3))
    # Verify antisymmetry: Omega should be equal to the negative of its transpose
    is_antisymmetric = jnp.allclose(kahler_matrix[0], -kahler_matrix[0].T)
    print(f"Is Omega antisymmetric? {is_antisymmetric}")
