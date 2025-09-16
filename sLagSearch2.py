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
newton_npts = 100000
newton_refine_steps = 100
#psi = 1000000
psi=0
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


coeffs_1e6 = jnp.asarray(
[[ 1.5693545e-02 , 0.0000000e+00,  0.0000000e+00,  8.8133715e-02,
  -1.5075313e-01 ,-5.3827003e-02, -5.9135702e-02,  1.6499399e-01,
   1.6606110e-01 , 2.8350067e-01, -4.6063700e-01,  5.9439220e-02,
  -7.1799718e-02 ,-2.9501135e-02, -3.0685129e-02,  1.9445103e-02,
  -1.8672526e-02 ,-1.0504410e-01, -2.4269110e-01, -8.3541858e-01,
  -2.1803735e-01 ,-1.6590050e-01, -6.1789149e-01,  1.0736270e-01,
  -1.6841620e-01],                                              
 [ 0.0000000e+00 , 2.0386826e-01,  0.0000000e+00,  9.4709508e-02,
   1.0768443e-01 , 9.5087759e-02,  4.1227022e-04, -1.0153750e-01,
  -7.6921709e-02 , 2.1377083e-02, -8.9596766e-01, -5.4174820e-03,
   7.9775631e-02 ,-2.7726306e-02, -1.5903161e-01, -1.5100638e-02,
   1.6243125e-01 , 3.5014410e-02,  2.0025412e-02, -9.8278038e-02,
  -1.5538788e-01 ,-1.7478925e-01, -6.8357980e-01,  2.3244463e-01,
  -4.5385453e-01],                                              
 [ 0.0000000e+00 , 0.0000000e+00,  6.1914021e-01, -1.4334874e-01,
  -1.1103801e-01 ,-3.9743548e-03,  1.7181499e-01, -5.8229398e-02,
  -1.9879566e-01 , 1.0774986e-01, -6.4248306e-01,  8.2803488e-02,
  -2.6878217e-01 , 3.4340270e-02, -3.5156373e-02,  1.2958262e-02,
  -8.8738188e-02 , 2.5129449e-01,  9.7601280e-02, -5.0516582e-01,
  -1.2623784e-02 ,-4.3266206e-03, -5.2790824e-02,  1.0808026e-04,
   6.3587010e-02]]
)

coeffs_slag = jnp.asarray(
[[0.027179092168807983, 0.0, 0.0, -0.07842623442411423, 0.10949830710887909, 0.07499706745147705, -0.14461739361286163, -0.005626048892736435, 0.044785548001527786, -0.18842148780822754, 0.16955344378948212, 0.03943122923374176, 0.22602003812789917, -0.009929579682648182, 0.011768237687647343, -0.9262008666992188, 0.04039764776825905, -0.49117445945739746, -0.226854145526886, 0.04022509604692459, -0.15865349769592285, 0.08033952862024307, -0.11675442010164261, -0.2145552933216095, -0.2871781885623932], [0.0, 0.13179196417331696, 0.0, 0.09516695141792297, -0.557324230670929, 0.08914212882518768, -0.044735703617334366, 0.06318096816539764, -0.4815204441547394, -0.1787245124578476, 0.030959367752075195, 0.16567638516426086, -0.08070563524961472, 0.002486596116796136, -0.013617032207548618, 0.5741927623748779, 0.0761655792593956, 0.2476537674665451, 0.13596521317958832, 0.012710952199995518, -0.0020229516085237265, 0.23357799649238586, -0.11952392011880875, -0.07694090157747269, 0.25460508465766907],
[0.0, 0.0, 0.012926698662340641, 0.172529399394989, 0.003992746118456125, -0.09851434826850891, 0.014487877488136292, -0.12042020261287689, -0.10701119154691696, -0.17210255563259125, 0.6347814202308655, -0.06003350019454956, 0.24548782408237457, -0.06856781244277954, 0.1329326331615448, 0.9371181726455688, 0.010400527156889439, 0.15061765909194946, 0.07388715445995331, 0.02970595471560955, 0.033107638359069824, 0.022720837965607643, 0.09464383870363235, -0.10430081188678741, 0.5139840245246887]]
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
total_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted_normalized, restriction, phases = compute_combined_fitness(min_set_real, coeffs, psi, debug_mode=True)
print('total_fitness: ', total_fitness)
print('lagrangian_fitness: ', lagrangian_fitness)
print('special_fitness: ', special_fitness)
print('Time to compute the total fitness', time.time() - st)

min_set = min_set_real[:,:5]+min_set_real[:,5:]*1j
CY_loss = jnp.sum(min_set**5, axis=1)
print(jnp.max(CY_loss), jnp.mean(CY_loss))
print(min_set_real[:,:5]+min_set_real[:,5:]*1j)
print(f"min_set_distance: Min: {jnp.min(distances)}, Max: {jnp.max(distances)}, Mean: {jnp.mean(distances)}")

with open("min_set_psi1000000.pkl", "wb") as f:
    pickle.dump(min_set, f)

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
plt.savefig(output_filename, dpi=300)

# --- 4. Close the Plot ---
# This prevents the plot from being displayed in a window,
# which is useful when running scripts automatically.
plt.close()

min_set_x1 = min_set_real[:, 2]
min_set_x2 = min_set_real[:, 3]

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
#for i in range(counts.shape[0]):
#    print(f"Phase: {phase_unique[i]}, Counts: {counts[i]}")
#print(min_set)
#
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
