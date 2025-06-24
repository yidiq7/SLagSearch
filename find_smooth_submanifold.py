import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
import numpy as np
from typing import Tuple, Union
from functools import partial
from collections import defaultdict

# Enable 64-bit precision for complex numbers
#jax.config.update("jax_enable_x64", True)
def combine_to_complex_equations(labels, coeffs):
    """
    Convert equations from Re/Im basis to zi*zjbar basis using:
    Im(zi*zjbar) = (zi*zjbar - zj*zibar)/(2i)
    Re(zi*zjbar) = (zi*zjbar + zj*zibar)/2
    
    Parameters:
    labels: array of strings like 'Im(z0*z1bar)', 'Re(z0*z1bar)'
    coeffs: 3x25 array of coefficients
    
    Returns:
    List of 3 strings representing the equations in terms of zi*zjbar
    """
    equations = []
    
    for eq_idx in range(coeffs.shape[0]):
        # Dictionary to store coefficients for each zi*zjbar term
        terms = defaultdict(complex)
        
        # Process each label and its coefficient
        for label_idx, label in enumerate(labels):
            coeff = coeffs[eq_idx, label_idx]
            
            # Skip if coefficient is very small
            if abs(coeff) < 1e-10:
                continue
            
            # Extract the zi and zj indices
            if label.startswith('Im(') and label.endswith(')'):
                # Extract 'zi*zjbar' from 'Im(zi*zjbar)'
                inner = label[3:-1]  # e.g., 'z0*z1bar'
                parts = inner.split('*')
                zi = parts[0]  # e.g., 'z0'
                zj = parts[1].replace('bar', '')  # e.g., 'z1'
                
                # Im(zi*zjbar) = (zi*zjbar - zj*zibar)/(2i)
                # Coefficient for zi*zjbar: coeff/(2i) = -coeff*i/2
                # Coefficient for zj*zibar: -coeff/(2i) = coeff*i/2
                
                terms[f"{zi}*{zj}bar"] += complex(0, -coeff/2)  # -i*coeff/2
                terms[f"{zj}*{zi}bar"] += complex(0, coeff/2)   # i*coeff/2
                
            elif label.startswith('Re(') and label.endswith(')'):
                # Extract 'zi*zjbar' from 'Re(zi*zjbar)'
                inner = label[3:-1]  # e.g., 'z0*z1bar'
                parts = inner.split('*')
                zi = parts[0]  # e.g., 'z0'
                zj = parts[1].replace('bar', '')  # e.g., 'z1'
                
                # Re(zi*zjbar) = (zi*zjbar + zj*zibar)/2
                # Coefficient for both zi*zjbar and zj*zibar: coeff/2
                
                terms[f"{zi}*{zj}bar"] += complex(coeff/2, 0)
                terms[f"{zj}*{zi}bar"] += complex(coeff/2, 0)
        
        # Build the equation string
        equation_parts = []
        for term in sorted(terms.keys()):
            coeff_complex = terms[term]
            
            # Skip if coefficient is essentially zero
            if abs(coeff_complex) < 1e-10:
                continue
            
            # Format the complex coefficient
            real_part = coeff_complex.real
            imag_part = coeff_complex.imag
            
            if abs(imag_part) < 1e-10:
                # Only real part
                coeff_str = f"{real_part:.6f}"
            elif abs(real_part) < 1e-10:
                # Only imaginary part
                if abs(imag_part - 1) < 1e-10:
                    coeff_str = "i"
                elif abs(imag_part + 1) < 1e-10:
                    coeff_str = "-i"
                else:
                    coeff_str = f"{imag_part:.6f}i"
            else:
                # Both real and imaginary parts
                if imag_part >= 0:
                    coeff_str = f"({real_part:.6f}+{imag_part:.6f}i)"
                else:
                    coeff_str = f"({real_part:.6f}{imag_part:.6f}i)"
            
            # Add to equation
            if equation_parts:
                if coeff_str.startswith('-'):
                    equation_parts.append(f" {coeff_str}*{term}")
                else:
                    equation_parts.append(f" + {coeff_str}*{term}")
            else:
                equation_parts.append(f"{coeff_str}*{term}")
        
        equation = "".join(equation_parts) + " = 0"
        equations.append(equation)
    
    return equations


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
    coeffs_normalized = coeffs / norms
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
    det = jnp.linalg.det(gram)
    
    # Return negative log of absolute determinant (minimize this = maximize determinant)
    return -jnp.log10(jnp.abs(det) + 1e-10)


@partial(jit, static_argnames=['k', 'lambda_reg'])
def loss_function(coeffs: jnp.ndarray, basis: jnp.ndarray, 
                  k: int = 10, lambda_reg: float = 1.0) -> float:

    # d is the order of the polynomial
    d = 1
    # Evaluate equations at all points
    coeffs = normalize_coeffs(coeffs)
    eq_values = evaluate_equations(coeffs, basis)
    
    # Inline k_smallest_sum logic
    eq_error = jnp.sqrt(jnp.sum(eq_values**(2/d), axis=1))
    sorted_values = jnp.sort(eq_error)
    intersect_loss = jnp.mean(sorted_values[:k])
    
    # Add linear independence penalty
    independence_penalty = compute_linear_independence_penalty(coeffs)
    
    # Total loss
    return intersect_loss + lambda_reg * independence_penalty


@partial(jit, static_argnames=['k'])
def loss_function_aray(coeffs: jnp.ndarray, basis: jnp.ndarray, 
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
                      init_coeffs: Union[jnp.ndarray, None] = None,
                      learning_rate: float = 0.01,
                      num_steps: int = 5000,
                      num_min_set: int = 500,
                      lambda_reg: float = 0.1,
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

    labels = np.array(get_basis_labels())
   
    if init_coeffs is None: 
        # Start with random initialization
        coeffs = jax.random.normal(key, (3, 25)) * 0.1
        coeffs = normalize_coeffs(coeffs) 
        # Add larger weights to imaginary cross terms (first 10 coefficients)
        # These are Im(zi*zjbar) for i < j
        #key, subkey = jax.random.split(key)
        #coeffs = coeffs.at[:, :10].add(jax.random.normal(subkey, (3, 10)) * 0.5)
    
    else:
        coeffs = init_coeffs
    
    # Setup optimizer with gradient clipping for stability
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.sgd(learning_rate)
    )
    opt_state = optimizer.init(coeffs)
    
    # Create loss and gradient functions
    #loss_fn = lambda c: 1/loss_function(c, basis, k=num_min_set, lambda_reg=0) + lambda_reg * compute_linear_independence_penalty(coeffs)
    loss_fn = lambda c: loss_function(c, basis, k=num_min_set, lambda_reg=lambda_reg)
    grad_fn = jit(grad(loss_fn))
    
    # Optimization loop
    losses = []
    for step in range(num_steps):
        loss_val = loss_fn(coeffs)
        losses.append(float(loss_val))
        
        if step % 10 == 0:
            independence_loss = lambda_reg * compute_linear_independence_penalty(coeffs)
            print(f"Step {step}, Loss: {loss_val:.6f}, Independence Loss: {independence_loss:.6f}")

            equations = combine_to_complex_equations(labels, coeffs)
            
            # Print the equations
            for i, eq in enumerate(equations):
                print(f"Equation {i+1}:")
                print(eq)
                print()
        
        # Compute gradients and update
        grads = grad_fn(coeffs)
        updates, opt_state = optimizer.update(grads, opt_state)
        coeffs = optax.apply_updates(coeffs, updates)
   
    coeffs = normalize_coeffs(coeffs) 

    return coeffs, losses


def find_satisfying_points(coeffs: jnp.ndarray, basis: jnp.ndarray, 
                           k: int = 10) -> jnp.ndarray:
    # Evaluate equations at all points
    eq_values = evaluate_equations(coeffs, basis)
    
    # Inline k_smallest_sum logic
    eq_squared = jnp.sum(eq_values**2, axis=1)
    sorted_values = jnp.sort(eq_squared)
    sorted_values_k = sorted_values[k]
    return jnp.where(eq_squared < sorted_values_k)[0]
    

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
