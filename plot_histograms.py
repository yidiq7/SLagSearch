from find_smooth_submanifold import *
from slag_condition import *
import jax
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import timeit
import os
from helper import canonicalize_coeffs, reconstruct_hermitian_matrices

jax.config.update('jax_default_matmul_precision', 'highest')
#with open('/projects/ruehlehet/yidi/sLag/data/50mil_patch0_3.pkl', 'rb') as f:
#    pts_50mil_patch0 = pickle.load(f)

newton_npts = 100000
newton_refine_steps = 100
psi = 0

#with open(f'/projects/ruehlehet/yidi/sLag/data_psi/5mil_patch0_psi{psi}_seed1024.pkl', 'rb') as f:
#with open('/projects/ruehlehet/yidi/sLag/data/5mil_patch0_1024.pkl', 'rb') as f:
with open('/projects/ruehlehet/yidi/sLag/data/5mil_patch0_343.pkl', 'rb') as f:
    pts_5mil_patch0 = pickle.load(f)

pts_5mil_patch0 = np.asarray(pts_5mil_patch0)

points_real = np.concatenate([np.real(pts_5mil_patch0), np.imag(pts_5mil_patch0)], axis=1)
points_real = jnp.asarray(points_real)

coeffs_RP3 = jnp.zeros((3, 25)).at[[0, 1, 2], [0, 1, 2]].set(1)
coeffs_T3 = jnp.zeros((3, 25)).at[[0, 1, 2], [10, 15, 19]].set(1).at[[0, 1, 2],[15, 19, 22]].set(-1)

coeffs_slag = jnp.asarray(
[[0.2426205426454544, 0.0, 0.0, 0.0007313763489946723, -0.2856019139289856, -0.20572295784950256, 0.18828649818897247, 0.029869092628359795, -0.09923622757196426, 0.22673745453357697, -0.17185665667057037, 0.2603021562099457, 0.241065114736557, -0.11289950460195541, -0.28617537021636963, -0.01636059209704399, 0.5398446917533875, -0.010966761969029903, 0.15876314043998718, 0.2536250054836273, 0.16446873545646667, 0.23974816501140594, -0.08775155246257782, 0.21533694863319397, -0.03343011438846588],
[0.0, 0.39446088671684265, 0.0, -0.07484503835439682, -0.053350046277046204, 0.1598605513572693, 0.3004281520843506, -0.2641794681549072, -0.07407970726490021, -0.08721709251403809, 0.22866259515285492, 0.15670965611934662, 0.1691352277994156, 0.10614511370658875, -0.11828554421663284, 0.1399996429681778, 0.22629377245903015, 0.08097495883703232, 0.27436280250549316, 0.5388283729553223, 0.27957823872566223, -0.11216448247432709, 0.22134777903556824, 0.33706241846084595, 0.17791129648685455],
[0.0, 0.0, 0.007120152935385704, 0.18530592322349548, -0.27944040298461914, -0.061079468578100204, 0.2805018126964569, -0.015957588329911232, -0.1878385841846466, 0.038394469767808914, 0.47294291853904724, 0.11168787628412247, 0.27657195925712585, -0.08994236588478088, -0.2744230628013611, 0.06558291614055634, 0.3579765260219574, 0.2891828417778015, 0.2435738742351532, 0.3995233178138733, 0.2511395812034607, 0.17810171842575073, 0.056694112718105316, 0.15984591841697693, 0.13332360982894897]]

)

coeffs_slag = jnp.asarray(
[[0.1465429663658142, 0.0, 0.0, 0.08876851201057434, -0.3057002127170563, -0.15337060391902924, 0.24742735922336578, 0.010434716939926147, -0.1494726985692978, 0.15525047481060028, 0.12162843346595764, 0.2104591429233551, 0.27743253111839294, -0.11106276512145996, -0.30365490913391113, 0.0213785283267498, 0.496697336435318, 0.13121606409549713, 0.2119934856891632, 0.34362414479255676, 0.21904580295085907, 0.22970089316368103, -0.02597855217754841, 0.20625482499599457, 0.043357353657484055],
[0.0, 0.3435303568840027, 0.0, -0.030188560485839844, -0.09895868599414825, 0.1274724155664444, 0.3139318525791168, -0.23269689083099365, -0.09973996877670288, -0.06859051436185837, 0.287862628698349, 0.15721985697746277, 0.1990278661251068, 0.07533629983663559, -0.1544533669948578, 0.13407422602176666, 0.26406577229499817, 0.12477962672710419, 0.28424501419067383, 0.5435811877250671, 0.2903870642185211, -0.06396955251693726, 0.20301347970962524, 0.3230874538421631, 0.17973263561725616],
[0.0, 0.0, 0.03048308752477169, 0.1852244883775711, -0.27931755781173706, -0.06105261668562889, 0.2803786098957062, -0.01595057174563408, -0.18775609135627747, 0.03837759047746658, 0.47273513674736023, 0.11163882166147232, 0.27645039558410645, -0.0899028480052948, -0.27430257201194763, 0.06555411219596863, 0.35781916975975037, 0.289055734872818, 0.24346698820590973, 0.399347722530365, 0.2510291039943695, 0.17802351713180542, 0.0566691979765892, 0.15977565944194794, 0.13326498866081238]]
)


perturbation_order = 0.05

seed = 1230
key = jax.random.PRNGKey(seed)
coeffs_random = jax.random.uniform(key, (3, 25), minval=-1, maxval=1)

#coeffs = coeffs_add_cond
#coeffs = coeffs_T3
#coeffs = coeffs_1e6_add_cond
coeffs = coeffs_slag
#coeffs = coeffs_1e6
#coeffs = coeffs_new
#coeffs = coeffs_random
#coeffs = coeffs_RP3 + perturbation_order * coeffs_random
coeffs  = canonicalize_coeffs(coeffs)
coeffs_slag = normalize_coeffs(coeffs)

coeffs_random =  canonicalize_coeffs(coeffs_random)
coeffs_random =  normalize_coeffs(coeffs_random)

seed = 1234
key = jax.random.PRNGKey(seed)
coeffs_random2 = jax.random.uniform(key, (3, 25), minval=-1, maxval=1)

coeffs_RP3 = coeffs_RP3 + perturbation_order * coeffs_random2
coeffs_RP3 = canonicalize_coeffs(coeffs_RP3)
coeffs_RP3 = normalize_coeffs(coeffs_RP3)

print('H: ', reconstruct_hermitian_matrices(coeffs_slag))

# Compute the average distance for random coeffs:
min_set_real, distances = filter_and_refine(points_real, coeffs_slag, psi, k=newton_npts, n_refine_steps=newton_refine_steps, constant_coord=0, debug_mode=True)
total_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted_new2, restriction_new2, phases_new2 = compute_combined_fitness(min_set_real, coeffs_slag, psi, debug_mode=True)

min_set_real, distances = filter_and_refine(points_real, coeffs_random, psi, k=newton_npts, n_refine_steps=newton_refine_steps, constant_coord=0, debug_mode=True)
total_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted_random, restriction_random, phases_random = compute_combined_fitness(min_set_real, coeffs_random, psi, debug_mode=True)

'''
min_set_real, distances = filter_and_refine(points_real, coeffs_RP3, psi, k=newton_npts, n_refine_steps=newton_refine_steps, constant_coord=0, debug_mode=True)
total_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted_RP3, phases_RP3= compute_combined_fitness(min_set_real, coeffs_RP3, psi, debug_mode=True)

idx_rp3 = jnp.where(jnp.abs(phases_RP3 < 0.1) | (jnp.abs(phases_RP3 - 2*jnp.pi) < 0.1))
'''
frobenius_norms_new2 = jnp.linalg.norm(kahler_form_restricted_new2, axis=(1, 2))
frobenius_norms_random = jnp.linalg.norm(kahler_form_restricted_random, axis=(1, 2))
#frobenius_norms_RP3 = jnp.linalg.norm(kahler_form_restricted_RP3, axis=(1, 2))[idx_rp3]
#sorted_norms = jnp.sort(frobenius_norms_RP3)
#norms_cut = sorted_norms[:int(sorted_norms.shape[0]*0.8)]

os.makedirs('plots_slag', exist_ok=True)

plt.figure(figsize=(10, 6))
plt.hist(frobenius_norms_new2, bins=200, alpha=0.7, label='Potential sLag', color='skyblue', density=True)
print('max norm:', jnp.max(frobenius_norms_new2))
plt.hist(frobenius_norms_random, bins=200, alpha=0.7, label='Random intersection', color='orange', density=True)
#plt.hist(norms_cut, bins=200, alpha=0.7, label='RP^3 with perturbation', color='#4CAF50', density=True)
plt.xlim(0, 1.5)
plt.ylim(0, 300)
plt.xlabel('Frobenius norm')
plt.ylabel('Counts')
plt.title('Distribution of the norm of the Kahler form')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('plots_slag/Kahler_form_loss_histogram.png')
plt.close()
'''
plt.figure(figsize=(10, 6))
plt.hist(phases_random, bins=200, alpha=0.7, label='Random intersection', color='orange', density=True)
plt.hist(phases_new2, bins=200, alpha=0.7, label='Potential sLag', color='skyblue', density=True)
plt.xlim(0, 6.28)
plt.xlabel('Phase')
plt.ylabel('Counts')
plt.title('Distribution of the phases of the holomorphic 3-form')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('plots_slag/phase_histogram_new2.png')
plt.close()
'''

number_of_bins = 1000
# Create the sub-plot with a polar projection
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

# Define the width of each bar
width = 2 * np.pi / number_of_bins

# --- Calculate counts first to determine the baseline ---
counts_A, bin_edges_A = np.histogram(phases_new2, bins=number_of_bins, range=(0, 2 * np.pi))
angles_A = bin_edges_A[:-1]

counts_B, bin_edges_B = np.histogram(phases_random, bins=number_of_bins, range=(0, 2 * np.pi))
angles_B = bin_edges_B[:-1]

#counts_C, bin_edges_C = np.histogram(phases_RP3[idx_rp3], bins=number_of_bins, range=(0, 2 * np.pi))
#angles_C = bin_edges_C[:-1]

peak_index = np.argmax(counts_A)
peak_angle = angles_A[peak_index]

# --- FIX 3: Set baseline dynamically to half the max peak height ---
max_count = counts_A.max()
baseline_radius = max_count / 2

# Plot the bars with the new baseline
ax.bar(angles_A, counts_A, width=width, alpha=0.7, color='skyblue', label='Potential sLag', bottom=baseline_radius)
ax.bar(angles_B, counts_B, width=width, alpha=0.7, color='orange', label='Random intersection', bottom=baseline_radius)
#ax.bar(angles_C, counts_C, width=width, alpha=0.7, color='#4CAF50', label='RP^3 with perturbation', bottom=baseline_radius)
outer_limit = baseline_radius + max_count * 1.05

#ax.plot([peak_angle, peak_angle + np.pi], [outer_limit, outer_limit], 'navy', linewidth=0.5, alpha=0.6, linestyle='--')

# --- Format the plot ---
ax.set_theta_zero_location('E')
ax.set_theta_direction(1)

# --- FIX 2: Set angle labels to RADIANS ---
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax.set_xticklabels(['0', 'π/2', 'π', '3π/2'], fontsize=12)

# --- FIX 1: Set a less dense, non-obtrusive grid ---
# Set radial grid lines at 75% and 100% of the max height
radial_grid_values = [baseline_radius + max_count * 0.25, baseline_radius + max_count * 0.5, baseline_radius + max_count*0.75]
ax.set_rgrids(radial_grid_values, angle=22.5)
ax.set_yticklabels([]) # Hide the number labels on the grid
ax.grid(True, linestyle='--', alpha=0.6) # Keep grid but make it faint

# Adjust the plot's outer limit to fit the data perfectly
ax.set_rlim(0, baseline_radius + max_count * 1.05)


ax.set_title('Distribution of the phases of the holomorphic 3-form', fontsize=16, pad=25)
ax.legend(bbox_to_anchor=(1.1, 1.05))

plt.savefig('plots_slag/circular_phase_histogram.png', bbox_inches='tight')

