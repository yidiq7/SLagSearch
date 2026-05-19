from find_smooth_submanifold import *
from slag_condition import *
import jax
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import timeit
from helper import assert_metric_psi_compatible, canonicalize_coeffs, convert_real_to_complex_batch, convert_real_to_complex_single, determine_patch_and_rescale_single, dwork_points_path

jax.config.update('jax_default_matmul_precision', 'highest')
#with open('/projects/ruehlehet/yidi/sLag/data/50mil_patch0_3.pkl', 'rb') as f:
#    pts_50mil_patch0 = pickle.load(f)
#newton_npts = 100000
#newton_refine_steps = 100
newton_npts = 10000
newton_refine_steps = 40
#psi = 1000000
psi = 0+0j
#metric = 'FS'
metric = 'k4_fermat'

assert_metric_psi_compatible(metric, psi)

# Edit this line if you're not using the Dwork-family naming convention,
# e.g. POINTS_FILE = "data/my_cicy.pkl"
POINTS_FILE = dwork_points_path(psi, seed=1024)

with open(POINTS_FILE, 'rb') as f:
    pts_5mil_patch0 = pickle.load(f)

pts_5mil_patch0 = np.asarray(pts_5mil_patch0)

points_real = np.concatenate([np.real(pts_5mil_patch0), np.imag(pts_5mil_patch0)], axis=1)
points_real = jnp.asarray(points_real)

coeffs_RP3 = jnp.zeros((3, 25)).at[[0, 1, 2], [0, 1, 2]].set(1)
coeffs_T3 = jnp.zeros((3, 25)).at[[0, 1, 2], [10, 15, 19]].set(1).at[[0, 1, 2],[15, 19, 22]].set(-1)
coeffs_new = jnp.asarray(
[[-4.42929119e-01,  7.00592697e-02, -4.86966133e-01, -4.08638567e-01,
  -4.67276126e-02,  1.27319157e-01,  1.41432479e-01,  4.49554622e-02,
  -8.29851404e-02, -2.06674740e-01, -1.01248916e-04, -4.98570651e-02,
   5.96929826e-02, -3.46238196e-01,  1.98963322e-02,  6.08862210e-05,
  -3.21559131e-01,  1.05372801e-01, -1.20193344e-02,  3.09847092e-05,
  -6.09012283e-02,  1.80263162e-01,  1.78690651e-04,  1.66079700e-01,
   2.22407491e-03],
 [ 5.16545713e-01, -1.91632226e-01, -1.26754537e-01,  3.68196726e-01,
   2.50224844e-02,  1.07329540e-01, -4.63079289e-02, -1.12049524e-02,
  -4.19456251e-02,  1.78805083e-01, -3.69236477e-05,  5.24779856e-02,
  -6.45594776e-01, -1.47399738e-01,  4.09226157e-02, -1.75023961e-05,
   9.43831503e-02,  1.22110561e-01,  6.47585914e-02,  3.88917033e-06,
  -3.78051139e-02,  5.27824312e-02,  3.75252647e-08, -1.46627873e-01,
  -1.91849449e-05],
 [-6.48345232e-01,  5.23517691e-02,  2.14412332e-01,  5.65616369e-01,
  -7.64850378e-02,  9.92875993e-02, -1.19586170e-01, -1.68957397e-01,
  -3.11606826e-04, -1.12161404e-02, -1.05903773e-05, -8.54518078e-03,
   2.32697263e-01,  1.15602165e-01,  3.08737792e-02,  1.12797068e-06,
  -1.94848970e-01,  1.67838652e-02,  1.03254896e-02, -7.77433343e-06,
   9.73734856e-02,  1.85498789e-01, -5.18850936e-03, -3.88839617e-02,
   3.00563383e-03]]

)


coeffs_slag = jnp.asarray(
[[0.09372571110725403, 0.0, 0.0, 0.04947570711374283, -0.27014032006263733, 0.290439248085022, -0.05873992294073105, -0.42513707280158997, -0.09717778861522675, 0.19989840686321259, 0.38664567470550537, -0.07062390446662903, 0.013155322521924973, -0.15705052018165588, 0.1234697476029396, 0.5427428483963013, -0.3402601480484009, -0.1772173047065735, 0.21868768334388733, 0.23365969955921173, -0.2153988480567932, 0.02255122922360897, 0.1117400974035263, -0.07398983091115952, 0.31562912464141846],
[0.0, 0.26131829619407654, 0.0, -0.24962270259857178, 0.6179611682891846, -0.263107031583786, -0.15143907070159912, 0.052236564457416534, 0.004278537351638079, -0.08185352385044098, 0.02838468924164772, -0.10935220867395401, -0.021405532956123352, -0.07216960936784744, 0.4299582839012146, -0.3281639814376831, 0.057126760482788086, 0.12651300430297852, -0.018878906965255737, 0.1969107687473297, -0.23213674128055573, -0.13236203789710999, -0.13546112179756165, -0.09763630479574203, 0.010455459356307983],
[0.0, 0.0, 0.1908203363418579, 0.014503059908747673, -0.06520576030015945, -0.028506482020020485, 0.018275123089551926, 0.27912238240242004, -0.13690531253814697, -0.2538205683231354, -0.25144749879837036, 0.2966477572917938, -0.18626874685287476, 0.0020461399108171463, -0.326728492975235, -0.22679422795772552, 0.4030601382255554, -0.0541677325963974, -0.37111616134643555, 0.12218055129051208, 0.3886817693710327, 0.0006958091980777681, 0.008297581225633621, 0.22004170715808868, -0.0043386672623455524]]
)

coeffs_slag = jnp.asarray(
[[0.008135111071169376, 0.0, 0.0, 0.034110456705093384, -0.051552366465330124, -0.09078127890825272, -0.21653585135936737, -0.10087347030639648, -0.036410510540008545, -0.09621314704418182, -0.10108081251382828, 0.01101162564009428, 0.8398250341415405, -0.12991644442081451, 0.04354538768529892, -0.12032864987850189, 0.022739004343748093, 0.252056360244751, -0.10091695934534073, -0.17623819410800934, -0.005732133984565735, 0.1443588137626648, -0.04366682097315788, 0.26619380712509155, -0.04954077675938606],
[0.0, 0.002528600161895156, 0.0, -0.11464469879865646, -0.016054080799221992, -0.032600075006484985, -0.012370459735393524, 0.012286623008549213, 0.027733193710446358, 0.035988904535770416, 0.13274334371089935, 0.052412550896406174, -0.9457160830497742, 0.16518470644950867, -0.02148539386689663, 0.07934409379959106, 0.0823899358510971, -0.0412033312022686, 0.020897284150123596, 0.19772838056087494, 0.016522396355867386, -0.0860254094004631, 0.0731077715754509, 0.03563130646944046, 0.11174309253692627],
[0.0, 0.0, 0.09465108066797256, 0.11126672476530075, 0.010422823950648308, 0.45290833711624146, 0.35332754254341125, 0.25843915343284607, 0.09951002895832062, 0.5054367184638977, -0.033156510442495346, 0.12722761929035187, -0.17186303436756134, 0.012448400259017944, -0.10432498902082443, 0.07591179758310318, 0.07454387843608856, 0.09396175295114517, -0.33415740728378296, 0.01807628944516182, -0.051696013659238815, -0.1775708943605423, 0.024721577763557434, -0.3077135682106018, 0.04064161330461502]]
)

perturbation_order = 0.001

seed = 1230
#seed = 42
key = jax.random.PRNGKey(seed)
coeffs_random = jax.random.uniform(key, (3, 25), minval=-1, maxval=1)

#coeffs = coeffs_new2
#coeffs = coeffs_new
#coeffs = coeffs_random
#coeffs = coeffs_RP3
#coeffs = coeffs_RP3 + perturbation_order * coeffs_random
#coeffs = coeffs_T3
#coeffs = coeffs_T3 + perturbation_order * coeffs_random
#coeffs = coeffs_1e6
#coeffs = coeffs_1e6_add_cond
coeffs = coeffs_slag

print('Original Coeffs: ', coeffs)
coeffs = canonicalize_coeffs(coeffs)
print('rref: ', coeffs)
coeffs = normalize_coeffs(coeffs)
print('normalized: ', coeffs)

coeffs_RP3 = normalize_coeffs(canonicalize_coeffs(coeffs_RP3))
coeffs_T3 = normalize_coeffs(canonicalize_coeffs(coeffs_T3))

def calculate_distance(ind1, ind2):
    return jnp.linalg.norm(ind1.ravel() - ind2.ravel())

print('RP3 to T3 distance: ', calculate_distance(coeffs_RP3, coeffs_T3))
print('RP3 to slag distance: ', calculate_distance(coeffs_RP3, coeffs))
print('T3 to slag distance: ', calculate_distance(coeffs, coeffs_T3))
print('random to slag distance: ', calculate_distance(coeffs_random, coeffs_T3))
print('random to RP3 distance: ', calculate_distance(coeffs_random, coeffs_RP3))

'''
# The symbolic restriction matrix X = sp.symbols('x0:5', real=True)
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
'''
# Compute the average distance for random coeffs:

st = time.time()
#min_set_real, distances = filter_and_refine(points_real, coeffs, psi, k=3000, n_refine_steps=5, debug_mode=True)
min_set_real, distances, _ = filter_and_refine(points_real, coeffs, psi, k=newton_npts, n_refine_steps=newton_refine_steps, n_repulsion_steps=20)
print('Finished Newton Method')
#min_set_real = filter_and_refine(points_real, coeffs, jacobian_func, psi, k=3000, n_refine_steps=5 
total_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted_normalized, restriction, phases = compute_combined_fitness(min_set_real, coeffs, psi, metric=metric, debug_mode=True)
print('total_fitness: ', total_fitness)
print('lagrangian_fitness: ', lagrangian_fitness)
print('special_fitness: ', special_fitness)
print('Time to compute the total fitness', time.time() - st)

min_set = convert_real_to_complex_batch(min_set_real)
CY_loss = jnp.sum(min_set**5, axis=1)
print(jnp.max(CY_loss), jnp.mean(CY_loss))
print(min_set_real[:,:5]+min_set_real[:,5:]*1j)
print(f"min_set_distance: Min: {jnp.min(distances)}, Max: {jnp.max(distances)}, Mean: {jnp.mean(distances)}")

#with open("min_set_psi0.pkl", "wb") as f:
#    pickle.dump(min_set, f)

frobenius_norms = jnp.linalg.norm(kahler_form_restricted_normalized, axis=(1, 2))
# Pick the smallest 90% to avoid numerical issues
sorted_norms = jnp.sort(frobenius_norms)
norms_cut = sorted_norms[:int(sorted_norms.shape[0]*0.99)]
print(f"Kahler loss: Min: {jnp.min(norms_cut)}, Max: {jnp.max(norms_cut)}, Mean: {jnp.mean(norms_cut)}")

values, indices = jax.lax.top_k(frobenius_norms, 10)
#print("Largest 20 values:", values)
#print("Indices of the largest 20 values:", indices)

#for i in range(50):
print('Point:', min_set[indices])
print('kahler_form_restricted: ', kahler_form_restricted_normalized[indices])
print('kahler_form_restricted norm: ', jnp.linalg.norm(kahler_form_restricted_normalized[indices], axis=(1,2)))
print('top k norm values', values)
print('restriction:', restriction[indices])

phase_tolerance = 1e-3
phase_rounded = jnp.round(phases / phase_tolerance) * phase_tolerance
phase_unique, counts = jnp.unique(phase_rounded, return_counts=True)
num_phases = len(phase_unique)

phases = phases % jnp.pi
counts, _ = jnp.histogram(phases, bins=100, range=(0, jnp.pi))
probs = counts / jnp.sum(counts)
epsilon = 1e-9
entropy = -jnp.sum(probs * jnp.log(probs + epsilon))
max_entropy = jnp.log(100)
print('phase histogram counts: ', counts)
print('entropy: ', entropy)
print('max_entropy: ', max_entropy)

plt.figure(figsize=(10, 6)) # Create a figure with a specific size for better quality
plt.hist(phases, bins=100, color='skyblue', edgecolor='black')

# Add titles and labels for clarity
plt.title('Histogram of Phase Array')
plt.xlabel('Phase Value (radians)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

# --- 3. Save the Histogram to a File ---
# The plot is saved to a PNG file in the same directory where the script is run.
# The `dpi` (dots per inch) argument can be adjusted to change the resolution of the saved image.
output_filename = 'phase_histogram.png'
#plt.savefig(output_filename, dpi=300)

# --- 4. Close the Plot ---
# This prevents the plot from being displayed in a window,
# which is useful when running scripts automatically.
plt.close()

'''
min_set_x1 = min_set_real[:, 3]
min_set_x2 = min_set_real[:, 2]

# We use Matplotlib's scatter function to plot y versus x.
plt.figure(figsize=(10, 6)) # Create a figure with a specific size for better quality
plt.scatter(min_set_x1, min_set_x2, alpha=0.6, color='purple', edgecolor='black',s=0.3)

# Add titles and labels for clarity
plt.title('Scatter Plot of x2 vs x1')
plt.xlabel('x1 values')
plt.ylabel('x2 values')
plt.grid(True, linestyle='--', alpha=0.6)

# --- 3. Save the Scatter Plot to a File ---
# The plot is saved to a PNG file in the same directory where the script is run.
# The `dpi` (dots per inch) argument can be adjusted to change the resolution of the saved image.
output_filename = 'scatter_plot.png'
plt.savefig(output_filename, dpi=300)

# --- 4. Close the Plot ---
# This prevents the plot from being displayed in a window,
# which is useful when running scripts automatically.
plt.close()
'''
print('First 1000 norms: ', norms_cut[:1000])
print('First 5000 norms: ', norms_cut[:5000])

distances_sorted = jnp.sort(distances)
print('First 5000 distances: ', distances_sorted[:5000])
print('Last 20 distances: ', distances_sorted[-20:])

PATCH_ACTIVE_INDICES = jnp.array([
    [1, 2, 3, 4, 6, 7, 8, 9],  # patch=0: skip 0,5
    [0, 2, 3, 4, 5, 7, 8, 9],  # patch=1: skip 1,6
    [0, 1, 3, 4, 5, 6, 8, 9],  # patch=2: skip 2,7
    [0, 1, 2, 4, 5, 6, 7, 9],  # patch=3: skip 3,8
    [0, 1, 2, 3, 5, 6, 7, 8],  # patch=4: skip 4,9
], dtype=jnp.int32)

from get_restriction import compute_affine_jacobian
from helper import (evaluate_equations_single_point, convert_complex_to_real_single,
                    convert_real_to_complex_single, determine_patch_and_rescale_single)

test_point = min_set_real[300]
p_complex = convert_real_to_complex_single(test_point)
p_complex_rescaled, patch_index = determine_patch_and_rescale_single(p_complex)

active_indices = PATCH_ACTIVE_INDICES[patch_index]

f_vec = evaluate_equations_single_point(test_point, coeffs, psi)
J = compute_affine_jacobian(test_point, patch_index, coeffs, psi)

print('f_vec: ', f_vec)
print('J: ', J)
JJT = J @ J.T + 1e-8 * jnp.eye(J.shape[0])
w = jnp.linalg.solve(JJT, -f_vec)
delta_p_active = J.T @ w

print('JJT: ', JJT)
print('w: ', w)
print('delta_p_active: ', delta_p_active)

test_point_new = test_point.at[active_indices].add(0.1*delta_p_active)

res = evaluate_equations_single_point(test_point, coeffs, psi)
res_new = evaluate_equations_single_point(test_point_new, coeffs, psi)

test_point_new = convert_real_to_complex_single(test_point_new)
test_point = convert_real_to_complex_single(test_point)

test_point_new_rescaled, patch_index = determine_patch_and_rescale_single(test_point_new)

print('before: ', test_point)
print('after: ', test_point_new)
print('after scaled: ', test_point_new_rescaled)
print('res_old: ', res)
print('res_new: ', res_new)

test_point_new_rescaled_real = convert_complex_to_real_single(test_point_new_rescaled)
res_new_rescaled = evaluate_equations_single_point(test_point_new_rescaled_real, coeffs, psi)
print('res_new_rescaled: ', res_new_rescaled)

