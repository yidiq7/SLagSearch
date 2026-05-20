import jax
import jax.numpy as jnp
from get_restriction import compute_affine_jacobian, compute_restriction, compute_Omega_restriction
from functools import partial
import math
from typing import Callable
from itertools import combinations_with_replacement
from collections import Counter
from helper import convert_real_to_complex_batch, determine_patches_batch, delete_index

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
    # Inhomogeneous coordinates (zeta) are the other 4 coordinates
    zeta = delete_index(z, patch_index)

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
    # Inhomogeneous coordinates (zeta) are the other 4 coordinates
    zeta = delete_index(z, patch_index)

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

# For degree=4 (k=4 Headrick-Nassar): 70 monomials, 5 canonical forms.
#   (5, 5) array of the unique, sorted exponent structures.
#   (70,) array of indices (0-4) mapping each monomial to its canonical form.
# For degree=5 (k=5 Headrick-Nassar): 126 monomials, 7 canonical forms.
#   (7, 5) array of the unique, sorted exponent structures.
#   (126,) array of indices (0-6) mapping each monomial to its canonical form.
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
    # The unique_coeffs array holds the Headrick-Nassar metric coefficients, one per
    # canonical exponent form. Inspect `_CANONICAL_EXPONENT_FORMS` for the ordering.
    #
    # For degree=4 (current): 5 canonical forms, 70 monomials.
    #   [0]: (1,1,1,1,0)  ->  5 monomials
    #   [1]: (2,1,1,0,0)  -> 30 monomials
    #   [2]: (2,2,0,0,0)  -> 10 monomials
    #   [3]: (3,1,0,0,0)  -> 20 monomials
    #   [4]: (4,0,0,0,0)  ->  5 monomials
    #
    # For degree=5: 7 canonical forms, 126 monomials.
    #   [0]: (1,1,1,1,1)  ->   1 monomial
    #   [1]: (2,1,1,1,0)  ->  20 monomials
    #   [2]: (2,2,1,0,0)  ->  30 monomials
    #   [3]: (3,1,1,0,0)  ->  20 monomials
    #   [4]: (3,2,0,0,0)  ->  20 monomials
    #   [5]: (4,1,0,0,0)  ->  20 monomials
    #   [6]: (5,0,0,0,0)  ->   5 monomials

    # Inhomogeneous coordinates (zeta) are the other 4 coordinates
    zeta = delete_index(z, patch_index)

    # Headrick-Nassar metric coefficients for the Fermat quintic.
    # degree=5 (commented out):
    #unique_coeffs = jnp.array([-4.79909*240, -75.298664*12, -83.726102*8, -103.669506*8, -39.049639*12, -33.852379*12, 1.0*(-180)])
    # degree=4 (active):
    unique_coeffs = jnp.array([2.214272*48, 14.010661*8, 2.3827940*24, 9.073280*12, 1.0*48])
    # Expand unique coefficients into the full coefficient array via gather.
    full_coeffs = unique_coeffs[_COEFF_MAPPING_INDICES]

    def kahler_potential(zeta: jnp.ndarray, zeta_bar: jnp.ndarray) -> float:
        """
        Defines K = log(sum_i gamma_i |m_i|^2).
        This function is structured to be compatible with jax.grad on complex vars.
        """
        # Re-homogenize both the holomorphic and anti-holomorphic coordinates
        z = jnp.insert(zeta, patch_index, 1.0)
        zbar = jnp.insert(zeta_bar, patch_index, 1.0)

        # Vectorize the monomial calculation over all 126 exponent sets.
        vmap_mono = jax.vmap(lambda exp, base: jnp.prod(base ** exp), in_axes=(0, None))
        
        # Calculate all 126 monomial values m_i(zeta)
        monomial_values = vmap_mono(_QUINTIC_EXPONENTS, z)
        
        # Calculate all 126 anti-monomial values m_i(zeta_bar)
        monomial_values_bar = vmap_mono(_QUINTIC_EXPONENTS, zbar)
        
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
    """Computes the real metric tensor G for a single point under `metric`."""
    if metric == 'FS':
        g_complex = calculate_complex_metric_FS(z, patch_index)
    elif metric == 'k4_fermat':
        g_complex = calculate_complex_metric_k4(z, patch_index)
    else:
        raise ValueError(f"Unsupported metric: '{metric}'")
    return _assemble_metric_tensor(g_complex)


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
    kahler_form = compute_kahler_form(points, patch_indices, metric)
    return jnp.einsum('nij,nik,njl->nkl', kahler_form, restriction, restriction)


def compute_kahler_form_unrestricted(
    points: jnp.ndarray,
    patch_indices: jnp.ndarray,
    metric: str = 'FS'
) -> jnp.ndarray:
    return compute_kahler_form(points, patch_indices, metric)


def compute_lagrangian_condition_fitness(kahler_form_unrestricted: jnp.ndarray, restriction: jnp.ndarray, k: int=10):
    kahler_form_restricted = jnp.einsum('nij,nik,njl->nkl', kahler_form_unrestricted, restriction, restriction)
    frobenius_norms = jnp.linalg.norm(kahler_form_restricted, axis=(1, 2))
    normalization_factor = jnp.linalg.norm(kahler_form_unrestricted, axis=(1, 2))
    norms_normalized = frobenius_norms / (normalization_factor + 1e-9)
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
    patch_indices: jnp.ndarray,
    psi: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the holomorphic 3-form accounting for different patches.
    
    The holomorphic form Ω changes between patches. In patch i, we have:
        Ω_i = (1/df/dz_i) × (wedge product of other coordinates)
    
    Args:
        points_complex: An (N, 5) complex array
        patch_indices: An (N,) integer array
        psi
        
    Returns:
        Omega: An (N,) complex array of the holomorphic form values
        Omega_min_indices: An (N,) integer array of which affine coordinate was used
        Omega_coord: An (N, 3) integer array of the three coordinates in the basis
    """
    # For each point, we work in its designated patch
    # The affine coordinates are all coordinates except patch_indices[i]
    def cy_hypersurface(z, psi):
        #return psi * jnp.prod(z)
        return jnp.sum(z**5) + 1 + psi * jnp.prod(z)

    dfdz_func = jax.jacrev(cy_hypersurface, argnums=0, holomorphic=True)
    
    def compute_omega_single(point: jnp.ndarray, patch_idx: int, psi: jnp.ndarray):
        """Compute Omega for a single point in a specific patch.

        Sign convention: an overall (-1)^max_idx factor.
          - (-1)^max_idx: Poincare residue. Moving dw_max_idx to the front of
            the ambient affine 4-form before taking the residue introduces
            (-1)^max_idx, which is needed for different max_idx choices within
            a patch to give the same Omega value.

        Note: the (-1)^patch_idx factor (Levi-Civita Euler form restriction)
        was previously included here. It was dropped because the phase of
        Omega|_L has an intrinsic +-1 gauge ambiguity (orientation of L)
        anyway; mod-pi fitness (compute_special_condition_fitness) treats
        theta and theta+pi as equivalent, making the per-patch sign moot.
        """
        # Get the 4 affine coordinates (excluding the patch coordinate)
        affine_coords = delete_index(point, patch_idx)

        dfdz = dfdz_func(affine_coords, psi)

        # Choose the one with the largest magnitude
        max_idx = jnp.argmax(jnp.abs(dfdz))
        sign = jnp.where(max_idx % 2 == 0, 1.0, -1.0)
        Omega = sign / dfdz[max_idx]

        # Get the coordinate indices in the affine system
        Omega_coord = get_Omega_coord(max_idx)

        return Omega, max_idx, Omega_coord
    
    vmapped_compute = jax.vmap(compute_omega_single, in_axes=(0, 0, None))
    Omega, Omega_min_indices, Omega_coord = vmapped_compute(points_complex, patch_indices, psi)
    
    return Omega, Omega_min_indices, Omega_coord


def compute_holomorphic_form_restricted(
    points_complex: jnp.ndarray,
    patch_indices: jnp.ndarray,
    psi: jnp.ndarray,
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
        If phase_only=True: An (N,) array of phases in [0, π)
        If phase_only=False: An (N,) complex array
    """
    Omega, Omega_min_indices, Omega_coord = compute_holomorphic_form(
        points_complex, patch_indices, psi
    )

    Omega_restriction = compute_Omega_restriction(restriction, Omega_coord)


    if phase_only:
        phase = jnp.angle(Omega * Omega_restriction)
        # Reduce mod pi: identify theta and theta+pi (sLag vs anti-sLag).
        phase = phase % jnp.pi

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
    Phase-concentration fitness on the half-circle.

    Phases are already reduced to [0, pi) by compute_holomorphic_form_restricted,
    so we histogram over the half-circle (theta identified with theta+pi). This
    counts sLag and anti-sLag together — appropriate when we only care about
    locating a Lagrangian whose Omega phase is constant up to sign. mod-pi is
    also the gauge-invariant quantity given the +-1 orientation ambiguity of
    Omega|_L (see compute_Omega_restriction).

    Args:
        phases: jnp.array of angles in [0, pi).
        n_bins: number of histogram bins on [0, pi).

    Returns:
        Scalar in [0, 1]. High fitness == high concentration in [0, pi).
    """
    # Create a histogram to approximate the distribution
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

@jax.jit
def compute_special_condition_fitness_smooth(phases: jnp.array) -> jnp.float32:
    """
    Differentiable fitness based on the Kuramoto order parameter on the half-circle.

    Phases are in [0, pi) (sLag and anti-sLag identified). We use exp(2i*theta)
    so that uniform-on-[0,pi) gives order parameter ~ 0 and perfectly concentrated
    phases give 1, matching the standard Kuramoto normalization.

    Returns:
        Scalar value in [0, 1].
        1.0 = perfectly identical phases mod pi (max concentration)
        0.0 = uniform distribution on [0, pi)
    """
    order_parameter = jnp.abs(jnp.mean(jnp.exp(2j * phases)))
    return order_parameter

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

    jacobians = vmap_compute_affine_jacobian(min_set_real, patch_indices, coeffs, psi)
    restrictions = vmap_compute_restriction(jacobians)

    kahler_form_unrestricted = compute_kahler_form_unrestricted(min_set, patch_indices, metric=metric)

    lagrangian_fitness = compute_lagrangian_condition_fitness(
        kahler_form_unrestricted, restrictions, k=10
    )

    phases = compute_holomorphic_form_restricted(
        min_set, patch_indices, psi, restrictions, phase_only=True
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
        #return combined_fitness, lagrangian_fitness, special_fitness, kahler_form_unrestricted, restrictions, phases
        return combined_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted_normalized, restrictions, phases
    else:
        return combined_fitness
