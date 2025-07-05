import jax
import jax.numpy as jnp

# --- Core Calculation ---

def _calculate_complex_metric(z: jnp.ndarray, patch_index: int) -> jnp.ndarray:
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

# --- Assembly Helpers ---

def _assemble_metric_tensor(g_complex: jnp.ndarray) -> jnp.ndarray:
    """Assembles the 8x8 real metric tensor G from the 4x4 complex metric."""
    g_real = jnp.real(g_complex)
    g_imag = jnp.imag(g_complex)
    # The real metric G is a block matrix: G = [[ R, -I ], [ I,  R ]]
    top_block = jnp.concatenate([g_real, -g_imag], axis=1)
    bottom_block = jnp.concatenate([g_imag, g_real], axis=1)
    return jnp.concatenate([top_block, bottom_block], axis=0)

def _assemble_kahler_form(g_complex: jnp.ndarray) -> jnp.ndarray:
    """Assembles the 8x8 real Kähler form Omega from the 4x4 complex metric."""
    g_real = jnp.real(g_complex)
    g_imag = jnp.imag(g_complex)
    # The Kähler form Omega = G*J, which results in: Omega = [[-I, -R], [R, -I]]
    top_block = jnp.concatenate([-g_imag, -g_real], axis=1)
    bottom_block = jnp.concatenate([g_real, -g_imag], axis=1)
    return jnp.concatenate([top_block, bottom_block], axis=0)

# --- Vmapped Single-Point Computers ---

def _compute_metric_for_single_point(z: jnp.ndarray, patch_index: int, epsilon: float) -> jnp.ndarray:
    """Computes the Fubini-Study metric tensor G for a single point."""
    return jax.lax.cond(
        jnp.abs(z[patch_index]) < epsilon,
        lambda: jnp.full((8, 8), jnp.nan, dtype=jnp.float32),
        lambda: _assemble_metric_tensor(_calculate_complex_metric(z, patch_index))
    )

def _compute_kahler_for_single_point(z: jnp.ndarray, patch_index: int, epsilon: float) -> jnp.ndarray:
    """Computes the Fubini-Study Kähler form Omega for a single point."""
    return jax.lax.cond(
        jnp.abs(z[patch_index]) < epsilon,
        lambda: jnp.full((8, 8), jnp.nan, dtype=jnp.float32),
        lambda: _assemble_kahler_form(_calculate_complex_metric(z, patch_index))
    )

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
        _compute_kahler_for_single_point, in_axes=(0, None, None)
    )
    return vmapped_computer(points_Z, patch_index, epsilon)


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
