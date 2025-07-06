from find_smooth_submanifold import *
from slag_condition import *
#from helper import *
import jax
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import timeit

jax.config.update('jax_default_matmul_precision', 'highest')
#with open('/projects/ruehlehet/yidi/sLag/data/50mil_patch0_3.pkl', 'rb') as f:
#    pts_50mil_patch0 = pickle.load(f)

with open('/projects/ruehlehet/yidi/sLag/data/5mil_patch0_343.pkl', 'rb') as f:
    pts_5mil_patch0 = pickle.load(f)

pts_5mil_patch0 = np.asarray(pts_5mil_patch0)

points_real = np.concatenate([np.real(pts_5mil_patch0), np.imag(pts_5mil_patch0)], axis=1)

coeffs_RP3 = jnp.zeros((3, 25)).at[[0, 1, 2], [0, 1, 2]].set(1)
#coeffs_T3 = jnp.zeros((3, 25)).at[[0, 1, 2], [10, 15, 19]].set(1).at[[0, 1, 2],[15, 19, 22]].set(-1)

perturbation_order = 0.0001

seed = 1230
key = jax.random.PRNGKey(seed)
coeffs_random = jax.random.uniform(key, (3, 25), minval=-1, maxval=1)

coeffs = coeffs_random
#coeffs = coeffs_RP3 + perturbation_order * coeffs_random
#coeffs = coeffs_T3 + perturbation_order * coeffs_random
coeffs = normalize_coeffs(coeffs)

# The symbolic restriction matrix
X = sp.symbols('x0:5', real=True)
Y = sp.symbols('y0:5', real=True)
# The order of symbols in XY is [x0,x1,x2,x3,x4, y0,y1,y2,y3,y4]
XY = X+Y

A_sym = sp.symbols('a0:25', real=True)
B_sym = sp.symbols('b0:25', real=True)
C_sym = sp.symbols('c0:25', real=True)

jacobian_sym = get_jacobian_symbolic(constant_coord=0)

# Plug in the coefficients
sub_dict = {
    **{A_sym[i]: coeffs[0, i] for i in range(25)},
    **{B_sym[i]: coeffs[1, i] for i in range(25)},
    **{C_sym[i]: coeffs[2, i] for i in range(25)},
}

jacobian_replaced = jacobian_sym.xreplace(sub_dict)
jacobian_func = sp.lambdify([XY], jacobian_replaced, 'jax')

# Compute the average distance for random coeffs:

#min_set_real = np.concatenate([np.real(min_set), np.imag(min_set)], axis=1)
min_set_real = filter_and_refine(points_real, coeffs, jacobian_func, k=3000, batch_size=1000000, n_refine_steps=5, constant_coord=0)
st = time.time()
total_fitness = compute_combined_fitness(min_set_real, coeffs, jacobian_func)
print('total_fitness: ', total_fitness)
print('Time to compute the total fitness', time.time() - st)




'''
# Create a new figure and axes for the plot.
fig, ax = plt.subplots(figsize=(12, 7))

# We use the bin_edges and counts to create a bar chart.
# The width of each bar is the difference between consecutive bin edges.
# We align the bars to the left edge of each bin.
bin_widths = jnp.diff(bin_edges)
ax.bar(bin_edges[:-1], hist_counts, width=bin_widths, align='edge', color='skyblue', edgecolor='black')

# Add labels and a title for clarity.
ax.set_xlabel("Frobenius Norm Value", fontsize=12)
ax.set_ylabel("Frequency (Count in Bin)", fontsize=12)
ax.set_title(f"Distribution of Frobenius Norms for the Kahler form loss", fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Define the output filename.
output_filename = "Kahler_form_loss_histogram_RP3.png"

# Save the figure to a file. The `dpi` (dots per inch) argument controls resolution.
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

# Close the plot to prevent it from being displayed in the environment (e.g., a Jupyter notebook).
plt.close(fig)
'''
