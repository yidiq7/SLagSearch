from find_smooth_submanifold import *
from slag_condition import *
import jax
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import timeit
from helper import canonicalize_coeffs

jax.config.update('jax_default_matmul_precision', 'highest')
#with open('/projects/ruehlehet/yidi/sLag/data/50mil_patch0_3.pkl', 'rb') as f:
#    pts_50mil_patch0 = pickle.load(f)
#newton_npts = 100000
#newton_refine_steps = 100
newton_npts = 10000
newton_refine_steps = 60
#psi = 1000000
psi=0
metric = 'k4_fermat'

with open('/projects/ruehlehet/yidi/sLag/data/5mil_patch0_343.pkl', 'rb') as f:
#with open(f'/projects/ruehlehet/yidi/sLag/data_psi/5mil_patch0_psi{psi}_seed1024.pkl', 'rb') as f:
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
[[0.11216665804386139, 0.0, 0.0, -0.02633090317249298, 0.36662667989730835, 0.026743261143565178, -0.16048716008663177, -0.01786438748240471, -0.20301805436611176, -0.019653748720884323, 0.23372647166252136, -0.1140589639544487, -0.10161467641592026, 0.5398538708686829, -0.04246379807591438, 0.22577781975269318, 0.1380372941493988, -0.16419844329357147, 0.18191319704055786, 0.10524267703294754, -0.20116859674453735, -0.24554167687892914, 0.3031361401081085, -0.06644229590892792, 0.5979450941085815],
[0.0, 0.0185030996799469, 0.0, -0.05686819553375244, 0.41180339455604553, -0.1612538844347, -0.03338941931724548, -0.06938966363668442, -0.2844865322113037, 0.03726506978273392, 0.2984565794467926, -0.14783278107643127, -0.06539043039083481, 0.3878329396247864, -0.11212556064128876, 0.19980256259441376, -0.08283418416976929, -0.11308067291975021, 0.1923774629831314, 0.20798492431640625, -0.29761967062950134, -0.23533309996128082, 0.4763703942298889, -0.20195569097995758, 0.406141996383667],
[0.0, 0.0, 0.1497003138065338, -0.0477476604282856, 0.39185118675231934, -0.21344517171382904, -0.06409186124801636, -0.04855991527438164, -0.2874813973903656, -0.01005624234676361, 0.2812081277370453, -0.1390218287706375, -0.039825718849897385, 0.3934180736541748, -0.10033828020095825, 0.2115786224603653, -0.12867453694343567, -0.12012004107236862, 0.20589330792427063, 0.16143667697906494, -0.20614327490329742, -0.255362868309021, 0.4688882827758789, -0.15645577013492584, 0.47072315216064453]]


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
#min_set_real, distances = filter_and_refine(points_real, coeffs, psi, k=3000, n_refine_steps=5, constant_coord=0, debug_mode=True)
min_set_real, distances = filter_and_refine(points_real, coeffs, psi, k=newton_npts, n_refine_steps=newton_refine_steps, constant_coord=0, debug_mode=True)
#min_set_real = filter_and_refine(points_real, coeffs, jacobian_func, psi, k=3000, n_refine_steps=5, constant_coord=0)
total_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted_normalized, restriction, phases = compute_combined_fitness(min_set_real, coeffs, psi, metric=metric, debug_mode=True)
print('total_fitness: ', total_fitness)
print('lagrangian_fitness: ', lagrangian_fitness)
print('special_fitness: ', special_fitness)
print('Time to compute the total fitness', time.time() - st)

min_set = min_set_real[:,:5]+min_set_real[:,5:]*1j
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
