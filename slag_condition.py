import jax
import jax.numpy as jnp
from get_restriction import compute_affine_jacobian, compute_restriction, compute_Omega_restriction
from functools import partial
import math
from itertools import combinations_with_replacement
from collections import Counter
from helper import convert_real_to_complex_batch, determine_patches_batch

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
    mask = jnp.arange(len(z)) != patch_index
    zeta = z[mask] / z_patch
    #zeta = jnp.delete(z, patch_index)

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
    mask = jnp.arange(len(z)) != patch_index
    zeta = z[mask] / z_patch

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

    mask = jnp.arange(len(z)) != patch_index
    zeta = z[mask] / z_patch

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

def _compute_metric_for_single_point(z: jnp.ndarray, patch_index: int, metric: str) -> jnp.ndarray:
    """Computes the Fubini-Study metric tensor G for a single point."""
    return _assemble_metric_tensor(calculate_complex_metric_k4(z, patch_index))


def compute_kahler_form(
    points_Z: jnp.ndarray,
    patch_indices: jnp.ndarray,
    metric: str = 'FS',
) -> jnp.ndarray:
    """
    Computes the Fubini-Study Kähler form Omega for points in CP^4.
    The Kähler form is a 2-form, represented here by its component matrix Omega_ij.
    This matrix is always antisymmetric.
    
    Args:
        points_Z: An (N, 5) array of complex numbers (homogeneous coordinates).
        patch_indices: An (N,) integer array specifying which patch each point is in
        metric: Either 'FS' or 'k4_fermat'
    
    Returns:
        An (N, 8, 8) array of real numbers. Each 8x8 matrix is the component
        matrix `Omega` of the Kähler form.
    """
    if metric == 'FS':
        metric_fn = calculate_complex_metric_FS
    elif metric == 'k4_fermat':
        metric_fn = calculate_complex_metric_k4
    else:
        raise ValueError(f"Unsupported metric: '{metric}'")
    
    def compute_single_point(z, patch_index):
        return _assemble_kahler_form(metric_fn(z, patch_index))
    
    vmapped_compute = jax.vmap(compute_single_point, in_axes=(0, 0))
    return vmapped_compute(points_Z, patch_indices)


def compute_kahler_form_restricted(
    points: jnp.ndarray, 
    restriction: jnp.ndarray, 
    patch_indices: jnp.ndarray, 
    metric: str = 'FS'
) -> jnp.ndarray:

    jit_compute_kahler_form = jax.jit(compute_kahler_form, static_argnums=(2,))
    kahler_form = jit_compute_kahler_form(points, patch_indices, metric)
    kahler_form_restricted = jnp.einsum('nij,nik,njl->nkl', kahler_form, restriction, restriction)
    return kahler_form_restricted


def compute_kahler_form_unrestricted(
    points: jnp.ndarray, 
    patch_indices: jnp.ndarray, 
    metric: str = 'FS'
) -> jnp.ndarray:

    jit_compute_kahler_form = jax.jit(compute_kahler_form, static_argnums=(2,))
    return jit_compute_kahler_form(points, patch_indices, metric)


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


def compute_holomorphic_form(
    points_complex: jnp.ndarray,
    patch_indices: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the holomorphic 3-form accounting for different patches.
    
    The holomorphic form Ω changes between patches. In patch i, we have:
        Ω_i = (constant/z_i^4) × (wedge product of other coordinates)
    
    Args:
        points_complex: An (N, 5) complex array
        patch_indices: An (N,) integer array
        
    Returns:
        Omega: An (N,) complex array of the holomorphic form values
        Omega_min_indices: An (N,) integer array of which affine coordinate was used
        Omega_coord: An (N, 3) integer array of the three coordinates in the basis
    """
    # For each point, we work in its designated patch
    # The affine coordinates are all coordinates except patch_indices[i]
    
    # Create affine coordinates by removing the patch coordinate
    # This is more complex since each point has a different patch
    
    def compute_omega_single(point: jnp.ndarray, patch_idx: int):
        """Compute Omega for a single point in a specific patch."""
        # Get the 4 affine coordinates (excluding the patch coordinate)
        affine_coords = jnp.delete(point, patch_idx)
        
        # Compute 1/(5*z_affine_i)^4 for each affine coordinate
        Omega_values = 1.0 / (5.0 * affine_coords**4)
        
        # Choose the one with smallest magnitude
        min_idx = jnp.argmin(jnp.abs(Omega_values))
        Omega = Omega_values[min_idx]
        
        # Get the coordinate indices in the affine system
        Omega_coord = get_Omega_coord(min_idx)
       
        # NOTE: I DONT THINK THIS IS NEEDED SINCE WE ALREADY REMOVED 
        # THE COORDINATES WHEN COMPUTING THE RESTRICTION BUT LETS DOUBLE
        # CHECK THIS. 
        # Adjust coordinate indices if they're >= patch_idx (since we removed one coordinate)
        # This remaps from affine indices back to original [0,1,2,3,4] indices
        #Omega_coord = jnp.where(Omega_coord >= patch_idx, Omega_coord + 1, Omega_coord)
        # Remove the patch coordinate from the list
        #Omega_coord = jnp.array([c for c in Omega_coord if c != patch_idx])[:3]
        
        return Omega, min_idx, Omega_coord
    
    vmapped_compute = jax.vmap(compute_omega_single)
    Omega, Omega_min_indices, Omega_coord = vmapped_compute(points_complex, patch_indices)
    
    return Omega, Omega_min_indices, Omega_coord


def compute_holomorphic_form_restricted(
    points_complex: jnp.ndarray,
    patch_indices: jnp.ndarray,
    restriction: jnp.ndarray,
    phase_only: bool
) -> jnp.ndarray:
    """
    Computes the restricted holomorphic form with patch-aware phase corrections.
    
    Args:
        points_complex: An (N, 5) complex array
        patch_indices: An (N,) integer array
        restriction: An (N, 8, 3) array
        phase_only: If True, return only phases; if False, return full Omega
        
    Returns:
        If phase_only=True: An (N,) array of phases in [0, 2π)
        If phase_only=False: An (N,) complex array
    """
    Omega, Omega_min_indices, Omega_coord = compute_holomorphic_form(
        points_complex, patch_indices
    )
    
    Omega_restriction = compute_Omega_restriction(restriction, Omega_coord)
    
    if phase_only:
        # Compute phase from the point's coordinate in its patch
        # Note: Omega_min_indices is relative to affine coordinates (0-3)
        # We need to map back to the original coordinate
        
        # For each point, get the actual coordinate index being used
        def get_actual_coord_idx(min_idx: int, patch_idx: int):
            """Map affine index back to homogeneous coordinate index."""
            # Affine coordinates are [0,1,2,3,4] with patch_idx removed
            affine_to_homo = jnp.arange(5)
            affine_to_homo = jnp.delete(affine_to_homo, patch_idx)
            return affine_to_homo[min_idx]
        
        actual_indices = jax.vmap(get_actual_coord_idx)(Omega_min_indices, patch_indices)
        
        # Get phases from the actual coordinates
        row_indices = jnp.arange(points_complex.shape[0])
        phase_Omega = -4 * jnp.angle(points_complex[row_indices, actual_indices])
        
        phase_restriction = jnp.angle(Omega_restriction)
        phase = phase_Omega + phase_restriction
        
        # Normalize to [0, 2π)
        phase = phase % (2 * jnp.pi)
        
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

vmap_compute_affine_jacobian = jax.vmap(compute_affine_jacobian, in_axes=(0, 0, None, None))
vmap_compute_restriction = jax.vmap(compute_restriction, in_axes=0)

def compute_combined_fitness(
    min_set_real: jnp.ndarray, 
    coeffs: jnp.ndarray, 
    psi: jnp.ndarray, 
    metric: str='FS', 
    debug_mode: bool=False
) -> jnp.float32:
    """
    Computes combined fitness with automatic patch detection and handling.
    
    Args:
        min_set_real: An (N, 10) array of points in real coordinates
        coeffs: A (3, 25) array of equation coefficients
        psi: Complex parameter for the quintic
        metric: 'FS' or 'k4_fermat'
        debug_mode: If True, return additional diagnostic information
        
    Returns:
        If debug_mode=False: fitness scalar
        If debug_mode=True: tuple of (fitness, lagrangian_fitness, special_fitness, 
                                      kahler_form_restricted, restriction, phases, patch_indices)
    """

    min_set = convert_real_to_complex_batch(min_set_real)
    patch_indices = determine_patches_batch(min_set) 
    print('patch_indices: ', patch_indices)
    jacobians = vmap_compute_affine_jacobian(min_set_real, patch_indices, coeffs, psi)
    restrictions = vmap_compute_restriction(jacobians)

    kahler_form_unrestricted = compute_kahler_form_unrestricted(min_set, patch_indices, metric=metric)

    lagrangian_fitness = compute_lagrangian_condition_fitness(
        kahler_form_unrestricted, restrictions, k=10
    )

    phases = compute_holomorphic_form_restricted(
        min_set, patch_indices, restrictions, phase_only=True
    )

    n_bins_val = 100
    special_fitness = compute_special_condition_fitness(phases, n_bins=n_bins_val)
   
    #combined_fitness = special_fitness
    combined_fitness = lagrangian_fitness * special_fitness 
    #combined_fitness = jnp.where(lagrangian_fitness > 0.98 , 1 + special_fitness, lagrangian_fitness)

    if debug_mode:
        kahler_form_restricted = compute_kahler_form_restricted(min_set, restrictions, patch_indices, metric=metric)
        normalization_factor = jnp.linalg.norm(kahler_form_unrestricted, axis=(1, 2))
        kahler_form_restricted_normalized = kahler_form_restricted / jnp.sqrt(normalization_factor[:, None, None])
        # Test
        #kahler_form_restricted_normalized = kahler_form_restricted 
        #kahler_form_unrestricted_normalized = compute_kahler_form_unrestricted(min_set, constant_coord=constant_coord)
        return combined_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted_normalized, restriction, phases
    else:
        return combined_fitness
