import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
import numpy as np
from typing import Tuple
from functools import partial

# Enable 64-bit precision for complex numbers
#jax.config.update("jax_enable_x64", True)

@jit
def generate_basis(points: jnp.ndarray) -> jnp.ndarray:
    """
    Generate basis functions from points on Fermat quintic.
    
    Args:
        points: (N, 5) complex array of points on the quintic
        
    Returns:
        basis: (N, 25) real array of basis functions
               First 10 are Im(zi*zjbar) for i<j, next 15 are Re(zi*zjbar) for i<=j
    """
    N = points.shape[0]
    
    # Create all pairwise products zi * zj_bar using broadcasting
    # Shape: (N, 5, 5)
    zi = points[:, :, None]  # (N, 5, 1)
    zj_bar = jnp.conj(points[:, None, :])  # (N, 1, 5)
    products = zi * zj_bar  # (N, 5, 5)
    
    # Extract upper triangular indices for imaginary parts (i < j)
    # This gives us 10 unique imaginary parts
    triu_indices_imag = jnp.triu_indices(5, k=1)
    imag_basis = jnp.imag(products[:, triu_indices_imag[0], triu_indices_imag[1]])  # (N, 10)
    
    # Extract upper triangular indices including diagonal for real parts (i <= j)
    # This gives us 15 unique real parts
    triu_indices_real = jnp.triu_indices(5, k=0)
    real_basis = jnp.real(products[:, triu_indices_real[0], triu_indices_real[1]])  # (N, 15)
    
    # Concatenate to form complete basis
    return jnp.concatenate([imag_basis, real_basis], axis=1)  # (N, 25)

@jit 
def normalize_coeffs(coeffs: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(coeffs, axis=1, keepdims=True)
    coeffs_normalized = coeffs / norm
    return coeffs_normalized

@jit
def evaluate_equations(coeffs: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate three equations at all points.
    
    Args:
        coeffs: (3, 25) array of coefficients for three equations
        basis: (N, 25) array of basis functions
        
    Returns:
        values: (N, 3) array of equation values at each point
    """
    return basis @ coeffs.T  # More efficient than jnp.dot

@jit
def compute_linear_independence_penalty(coeffs: jnp.ndarray) -> float:
    """
    Compute penalty for linear dependence of equations.
    Uses the determinant of the Gram matrix.
    
    Args:
        coeffs: (3, 25) array of coefficients
        
    Returns:
        penalty: scalar penalty (higher when equations are more dependent)
    """
    # Compute Gram matrix
    gram = coeffs @ coeffs.T
    
    # Regularized log determinant to avoid numerical issues
    det = jnp.linalg.det(gram + 1e-8 * jnp.eye(3))
    
    # Return negative log of absolute determinant (minimize this = maximize determinant)
    return -jnp.log(jnp.abs(det) + 1e-10)

@partial(jit, static_argnames=['k', 'lambda_reg'])
def loss_function(coeffs: jnp.ndarray, basis: jnp.ndarray, 
                  k: int = 10, lambda_reg: float = 1.0) -> float:
    # Evaluate equations at all points
    eq_values = evaluate_equations(coeffs, basis)
    
    # Inline k_smallest_sum logic
    eq_squared = jnp.sum(eq_values**2, axis=1)
    sorted_values = jnp.sort(eq_squared)
    intersect_loss = jnp.sum(sorted_values[:k])
    
    # Add linear independence penalty
    independence_penalty = compute_linear_independence_penalty(coeffs)
    
    # Total loss
    return intersect_loss + lambda_reg * independence_penalty

@partial(jit, static_argnames=['k', 'lambda_reg'])
def loss_function(coeffs: jnp.ndarray, basis: jnp.ndarray, 
                  k: int = 10, lambda_reg: float = 1.0) -> float:
    # Evaluate equations at all points
    eq_values = evaluate_equations(coeffs, basis)
    
    # Inline k_smallest_sum logic
    eq_squared = jnp.sum(eq_values**2, axis=1)
    sorted_values = jnp.sort(eq_squared)
    intersect_loss = jnp.sum(sorted_values[:k])
    
    # Add linear independence penalty
    independence_penalty = compute_linear_independence_penalty(coeffs)
    
    # Total loss
    return intersect_loss + lambda_reg * independence_penalty


@partial(jit, static_argnames=['k'])
def loss_function_array(coeffs: jnp.ndarray, basis: jnp.ndarray, 
                  k: int = 10) -> jnp.ndarray:
    # Evaluate equations at all points
    eq_values = evaluate_equations(coeffs, basis)
    
    # Inline k_smallest_sum logic
    eq_squared = jnp.sum(eq_values**2, axis=1)
    sorted_values = jnp.sort(eq_squared)
    intersect_loss = sorted_values[:k]
    
    # Total loss
    return intersect_loss


def get_basis_labels():
    """Get human-readable labels for basis functions."""
    labels = []
    
    # Imaginary parts for i < j
    for i in range(5):
        for j in range(i+1, 5):
            labels.append(f"Im(z{i}*z{j}bar)")
    
    # Real parts for i <= j
    for i in range(5):
        for j in range(i, 5):
            labels.append(f"Re(z{i}*z{j}bar)")
    
    return labels

def optimize_equations(points: jnp.ndarray, 
                      learning_rate: float = 0.01,
                      num_steps: int = 5000,
                      lambda_reg: float = 1.0,
                      seed: int = 42) -> Tuple[jnp.ndarray, list]:
    """
    Optimize coefficients for three equations.
    
    Args:
        points: (N, 5) complex array of points on Fermat quintic
        learning_rate: optimizer learning rate
        num_steps: number of optimization steps
        lambda_reg: regularization weight for linear independence
        seed: random seed
        
    Returns:
        coeffs: (3, 25) optimized coefficients
        losses: list of loss values during optimization
    """
    # Generate basis functions
    basis = generate_basis(points)
    
    # Initialize coefficients with focus on imaginary cross terms
    key = jax.random.PRNGKey(seed)
    
    # Start with random initialization
    coeffs = jax.random.normal(key, (3, 25)) * 0.1
    
    # Add larger weights to imaginary cross terms (first 10 coefficients)
    # These are Im(zi*zjbar) for i < j
    key, subkey = jax.random.split(key)
    coeffs = coeffs.at[:, :10].add(jax.random.normal(subkey, (3, 10)) * 0.5)
    
    # Setup optimizer with gradient clipping for stability
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(coeffs)
    
    # Create loss and gradient functions
    loss_fn = lambda c: loss_function(c, basis, lambda_reg)
    grad_fn = jit(grad(loss_fn))
    
    # Optimization loop
    losses = []
    for step in range(num_steps):
        loss_val = loss_fn(coeffs)
        losses.append(float(loss_val))
        
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss_val:.6f}")
        
        # Compute gradients and update
        grads = grad_fn(coeffs)
        updates, opt_state = optimizer.update(grads, opt_state)
        coeffs = optax.apply_updates(coeffs, updates)
    
    return coeffs, losses

def find_satisfying_points(coeffs: jnp.ndarray, points: jnp.ndarray, tolerance: float = 1e-6) -> jnp.ndarray:
    """Find indices of points that satisfy all three equations."""
    basis = generate_basis(points)
    eq_values = evaluate_equations(coeffs, basis)
    eq_squared = jnp.sum(eq_values**2, axis=1)
    return jnp.where(eq_squared < tolerance)[0]

def analyze_solution(coeffs: jnp.ndarray, points: jnp.ndarray) -> None:
    """
    Analyze the optimized solution.
    
    Args:
        coeffs: (3, 25) optimized coefficients
        points: (N, 5) complex array of points
    """
    basis = generate_basis(points)
    eq_values = evaluate_equations(coeffs, basis)
    eq_squared = jnp.sum(eq_values**2, axis=1)
    
    # Find points that satisfy all equations
    tolerance = 1e-6
    satisfying_indices = find_satisfying_points(coeffs, points, tolerance)
    satisfying_count = len(satisfying_indices)
    
    print(f"\nSolution Analysis:")
    print(f"Number of points satisfying all equations: {satisfying_count}")
    print(f"Minimum equation residual: {jnp.sqrt(jnp.min(eq_squared)):.10f}")
    
    # Compute condition number of coefficient matrix for stability analysis
    gram = coeffs @ coeffs.T
    eigenvalues = jnp.linalg.eigvalsh(gram)
    condition_number = jnp.max(eigenvalues) / jnp.min(eigenvalues)
    print(f"Condition number of equation system: {condition_number:.2f}")
    print(f"Linear independence measure (det of Gram matrix): {jnp.linalg.det(gram):.6f}")
    
    # Get basis labels
    labels = get_basis_labels()
    
    # Print dominant coefficients for each equation
    for eq_idx in range(3):
        print(f"\nEquation {eq_idx + 1} dominant terms:")
        abs_coeffs = jnp.abs(coeffs[eq_idx])
        top_indices = jnp.argsort(abs_coeffs)[-5:][::-1]
        
        for idx in top_indices:
            print(f"  {labels[idx]}: {coeffs[eq_idx, idx]:+.4f}")
    
    # If we found satisfying points, analyze them
    if satisfying_count > 0:
        print(f"\nAnalyzing {min(5, satisfying_count)} satisfying points:")
        for i in range(min(5, satisfying_count)):
            pt_idx = satisfying_indices[i]
            pt = points[pt_idx]
            print(f"  Point {pt_idx}: |z0|={jnp.abs(pt[0]):.3f}, "
                  f"|z1|={jnp.abs(pt[1]):.3f}, |z2|={jnp.abs(pt[2]):.3f}, "
                  f"|z3|={jnp.abs(pt[3]):.3f}, |z4|={jnp.abs(pt[4]):.3f}")
