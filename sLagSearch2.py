from find_smooth_submanifold import *
from get_restriction import *
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
'''
with open('fermat_rp3_points.pkl', 'rb') as f:
    min_set = pickle.load(f)

min_set = jnp.asarray(min_set['points']['complex'])
#print(min_set)
'''
#with open('/projects/ruehlehet/yidi/sLag/data/50mil_patch0_3.pkl', 'rb') as f:
#    pts_50mil_patch0 = pickle.load(f)

with open('/projects/ruehlehet/yidi/sLag/data/5mil_patch0_343.pkl', 'rb') as f:
    pts_5mil_patch0 = pickle.load(f)
pts_5mil_patch0 = np.asarray(pts_5mil_patch0)

#pts_5mil_patch0 = np.asarray(pts_50mil_patch0)

points_real = np.concatenate([np.real(pts_5mil_patch0), np.imag(pts_5mil_patch0)], axis=1)

coeffs_RP3 = jnp.zeros((3, 25)).at[[0, 1, 2], [0, 1, 2]].set(1)
coeffs_T3 = jnp.zeros((3, 25)).at[[0, 1, 2], [10, 15, 19]].set(1).at[[0, 1, 2],[15, 19, 22]].set(-1)

perturbation_order = 0

seed = 123
key = jax.random.PRNGKey(seed)
coeffs_random = jax.random.uniform(key, (3, 25), minval=-1, maxval=1)

#coeffs = coeffs_random
#coeffs = coeffs_RP3 + perturbation_order * coeffs_random
coeffs = coeffs_T3 + perturbation_order * coeffs_random
coeffs = normalize_coeffs(coeffs)

# The symbolic restriction matrix
jacobian_sym = get_jacobian_symbolic(constant_coord=0)
X = sp.symbols('x0:5', real=True)
Y = sp.symbols('y0:5', real=True)
# The order of symbols in XY is [x0,x1,x2,x3,x4, y0,y1,y2,y3,y4]
XY = X+Y

A_sym = sp.symbols('a0:25', real=True)
B_sym = sp.symbols('b0:25', real=True)
C_sym = sp.symbols('c0:25', real=True)

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
st = time.time()
min_set_real = filter_and_refine(points_real, coeffs, jacobian_func, k=3000, batch_size=1000000, n_refine_steps=5, constant_coord=0)
min_set = min_set_real[:, :5] + 1j*min_set_real[:, 5:]
print('Time to find the min set: ', time.time() - st)

st = time.time()
jacobian_func_batched = jax.vmap(jacobian_func, in_axes=0)
jacobian = jacobian_func_batched(min_set_real)
restriction = jax.jit(jax.vmap(get_restriction, in_axes=0))(jacobian)

jit_compute_kahler_form = jax.jit(compute_kahler_form, static_argnums=(1,))
kahler_form = jit_compute_kahler_form(min_set, 0)

kahler_form_restricted = jnp.einsum('nij,nik,njl->nkl', kahler_form, restriction, restriction)

frobenius_norms = jnp.linalg.norm(kahler_form_restricted, axis=(1, 2))
# Pick the smallest 90% to avoid numerical issues
sorted_norms = jnp.sort(frobenius_norms)
norms_cut = sorted_norms[:int(len(sorted_norms)*0.9)]
kahler_form_loss = jnp.mean(norms_cut)
print('Average norm of the Kahler form: ', kahler_form_loss)
hist_counts, bin_edges = jnp.histogram(norms_cut, bins=100)
#print('Bin Edges: ', bin_edges)
#print('Counts: ', hist_counts)
indices = jnp.argpartition(frobenius_norms, -1)[-1:]
print(frobenius_norms[indices])
print(min_set[indices])
print(kahler_form_restricted[indices])
print('Time to compute the kahler form loss: ', time.time() - st)

st = time.time()
Omega = 1 / (5*min_set[:, 1:])**4
# Pick the smallest one to avoid numeric issues
Omega_min_indices = jnp.argmin(jnp.abs(Omega), axis=1)
Omega = Omega[jnp.arange(Omega.shape[0]), Omega_min_indices]

def get_Omega_coord(min_idx):
    # Choose the rest three coordinates to form the basis
    coord_lookup_table = jnp.array([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2]
    ])
    return coord_lookup_table[min_idx]

Omega_coord = jax.vmap(get_Omega_coord)(Omega_min_indices)

# Setting z_4 as the dependent coordinate
Omega_restriction = compute_Omega_restriction(restriction, Omega_coord)
#Omega_restricted = Omega * Omega_restriction
phase_Omega = -4*jnp.angle(min_set[jnp.arange(min_set.shape[0]), Omega_min_indices+1])
#phase_Omega = jnp.angle(Omega)
phase_restriction = jnp.angle(Omega_restriction)
phase = phase_Omega + phase_restriction
phase = phase % jnp.pi
#phase = jnp.angle(Omega_restricted)
phase_tolerance = 1e-2
phase_rounded = jnp.round(phase / phase_tolerance) * phase_tolerance
phase_unique, counts = jnp.unique(phase_rounded, return_counts=True)

num_phases = len(counts[counts > 50])
unique_phases = phase_unique[counts > 50]
#num_phases = len(phase_unique)
print('Number of phases:', num_phases)
print('Unique phases: ', unique_phases)
print('Time to compute the phases of the 3-form', time.time() - st)

'''
for i in [14]:
#for i in range(40):
    #print(f'Omega: {Omega[i]}, Omega_restricted: {Omega_restricted[i]}, det: {Omega_restriction[i]}, Phase: {phase[i]}, Omega_min_indices: {Omega_min_indices[i]}, Points: {min_set[i]}')
    print(f'Omega: {Omega[i]},  det: {Omega_restriction[i]}, phase_Omega: {phase_Omega[i]}, phase_restriction: {phase_restriction[i]}, Phase: {phase[i]}, Omega_min_indices: {Omega_min_indices[i]}, Points: {min_set_real[i]}')
    print(restriction[i])
    print('--------------')

'''
'''
for i in range(counts.shape[0]):
    print(f"Phase: {phase_unique[i]}, Counts: {counts[i]}")
print(min_set[:10])
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
