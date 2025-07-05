import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
import numpy as np
from typing import Tuple, Union, Callable
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
    # We normalize on the complex basis zizjbar instead of the real and imaginary
    # parts. So we rescale the real part of zizibar by 1/sqrt(2) to get the correct
    # normalization since they are only counted once instead of twice compared to
    # the upper triangular terms.
    zzbar_indices = jnp.array([10, 15, 19, 22, 24])
    coeffs = coeffs.at[:,zzbar_indices].divide(jnp.sqrt(2.0))
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


# ------------------------------------------------------------------------------
# New loss function

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

def evaluate_equations_single_point(point: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the five equations. The input points are real."""
    point_complex = point[:5] + 1j * point[5:]
    basis = generate_basis_single_point(point_complex)
    eqs_vec = coeffs @ basis # (3,)
    cy = jnp.sum(point_complex**5)
    eqs_evaluated = jnp.array([jnp.real(cy), jnp.imag(cy), *eqs_vec]) 
    return eqs_evaluated

def approx_distance_newton_step(
    p_10d: jnp.ndarray, coeffs: jnp.ndarray, jacobian_func: Callable, constant_coord: int
) -> float:
    """Raw single-item function for computing the norm of a Newton step."""
    f_vec = evaluate_equations_single_point(p_10d, coeffs)
    J = jacobian_func(p_10d)
    JJT = J @ J.T + 1e-8 * jnp.eye(J.shape[0])
    w = jnp.linalg.solve(JJT, -f_vec)
    delta_p_active = J.T @ w
    return jnp.linalg.norm(delta_p_active)


def refine_point_iterative(
    p_10d_initial: jnp.ndarray, coeffs: jnp.ndarray, jacobian_func: Callable, constant_coord: int, n_steps: int
) -> jnp.ndarray:
    """Raw single-item function for refining a point."""
    active_indices = jnp.concatenate([
                         jnp.arange(0, constant_coord),
                         jnp.arange(constant_coord + 1, constant_coord + 5),
                         jnp.arange(constant_coord + 6, 10)
                     ])

    def body_fn(i, p_10d):
        f_vec = evaluate_equations_single_point(p_10d, coeffs)
        J = jacobian_func(p_10d)
        JJT = J @ J.T + 1e-8 * jnp.eye(J.shape[0])
        w = jnp.linalg.solve(JJT, -f_vec)
        delta_p_active = J.T @ w
        #jax.debug.print("Iteration {i}: delta_p_active = {x}, f_vec = {f_vec}, J = {J}", i=i, x=delta_p_active, f_vec=f_vec, J=J)
        return p_10d.at[active_indices].add(delta_p_active)

    return jax.lax.fori_loop(0, n_steps, body_fn, p_10d_initial)


def compute_distances_batched(points: np.ndarray, coeffs: jnp.ndarray, jacobian_func: Callable, batch_size: int = 100000, constant_coord: int = 0) -> jnp.ndarray:
    """ Compute the distances of the input points to the intersection"""

    num_points = points.shape[0]
    num_batches = (num_points + batch_size - 1) // batch_size

    # --- Create efficient batched functions that use your provided jacobian_func ---
    # We use partial to "bake in" the jacobian_func for the calls inside vmap.
    dist_partial = partial(approx_distance_newton_step, coeffs=coeffs, jacobian_func=jacobian_func, constant_coord=constant_coord)

    compute_distances = jax.jit(jax.vmap(dist_partial))
    # ---

    all_distances = None

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_points)
        batch_10d = jnp.array(points[start_idx:end_idx])

        distances = compute_distances(batch_10d)
        if all_distances is None:
            all_distances = distances
        else:
            all_distances = np.concatenate([all_distances, distances])

    return all_distances


def filter_and_refine(
    points: np.ndarray,
    coeffs: jnp.ndarray,
    jacobian_func: Callable,
    k: int = 10000,
    batch_size: int = 100000,
    n_refine_steps: int = 20,
    constant_coord: int = 0
) -> jnp.ndarray:
    """
    Filters a large set of 10D points to find the k best, then refines them.
    """

    refine_partial = partial(refine_point_iterative, coeffs=coeffs, jacobian_func=jacobian_func, constant_coord=constant_coord, n_steps=n_refine_steps)
    refine_batch = jax.jit(jax.vmap(refine_partial))

    all_distances = compute_distances_batched(points, coeffs, jacobian_func, batch_size=batch_size, constant_coord=constant_coord)

    best_2k_indices = np.argsort(all_distances)[:2*k]
    top_2k_points = points[best_2k_indices]

    best_k_indices = np.argsort(all_distances)[:k]
    top_k_points = points[best_k_indices]

    print(f"\nFound {k} best candidates. Refining them with {n_refine_steps} Newton steps...")

    distance_initial = compute_distances_batched(top_k_points, coeffs, jacobian_func, batch_size=k, constant_coord=constant_coord)
    refined_points_10d = refine_batch(jnp.array(top_2k_points))
    distance_refined = compute_distances_batched(refined_points_10d, coeffs, jacobian_func, batch_size=k, constant_coord=constant_coord)

    best_indices = np.argsort(distance_refined)[:k]
    top_k_distances = distance_refined[best_indices]
    top_k_points = refined_points_10d[best_indices]
    print(f"Refinement complete. Initial Distance range: {np.min(distance_initial)} ~ {np.max(distance_initial)}, Mean: {np.mean(distance_initial)}")
    print(f"Refined Distance range: {np.min(top_k_distances)} ~ {np.max(top_k_distances)}, Mean: {np.mean(top_k_distances)}")

    return top_k_points
