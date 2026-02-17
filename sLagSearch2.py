from find_smooth_submanifold import *
from slag_condition import *
import jax
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import timeit
from helper import canonicalize_coeffs, convert_real_to_complex_batch, convert_real_to_complex_single, determine_patch_and_rescale_single
from plots import make_fitness_plots

jax.config.update('jax_default_matmul_precision', 'highest')
#with open('/projects/ruehlehet/yidi/sLag/data/50mil_patch0_3.pkl', 'rb') as f:
#    pts_50mil_patch0 = pickle.load(f)
#newton_npts = 100000
#newton_refine_steps = 100
newton_npts = 10000
newton_refine_steps = 100
#psi = 10000000
psi=0
#psi=100
#metric = 'FS'
metric = 'k4_fermat'

#with open('/projects/ruehlehet/yidi/sLag/data/5mil_patch0_343.pkl', 'rb') as f:
#with open(f'/projects/ruehlehet/yidi/sLag/data_psi/5mil_patch0_psi{psi}_seed1024.pkl', 'rb') as f:
with open(f'/projects/ruehlehet/yidi/sLag/data_psi/1mil_patch_all_psi{psi}_seed1024.pkl', 'rb') as f:
    pts_5mil_patch0 = pickle.load(f)

pts_5mil_patch0 = np.asarray(pts_5mil_patch0)

points_real = np.concatenate([np.real(pts_5mil_patch0), np.imag(pts_5mil_patch0)], axis=1)
points_real = jnp.asarray(points_real)

coeffs_RP3 = jnp.zeros((3, 25)).at[[0, 1, 2], [0, 1, 2]].set(1)
coeffs_T3 = jnp.zeros((3, 25)).at[[0, 1, 2], [10, 15, 19]].set(1).at[[0, 1, 2],[15, 19, 22]].set(-1)
#Jcoeffs_slag = jnp.asarray([
#        [-0.0367027148604393, 0.0025424568448215723, -0.21919819712638855, 0.459602952003479, -0.14038687944412231, 0.17836026847362518, -0.02632879465818405, 0.1542496383190155, -0.21669824421405792, 0.08787675201892853, -0.02216939628124237, 0.4584207534790039, 0.18183615803718567, 0.006892359349876642, 0.09927520900964737, -0.019934242591261864, -0.16006404161453247, -0.1488436758518219, 0.4862450659275055, 0.07581081986427307, -0.1970602571964264, 0.04495194926857948, 0.09229589253664017, 0.15134599804878235, 0.005224315449595451],
#        [0.42350825667381287, 0.16512976586818695, -0.009587904438376427, 0.020616931840777397, -0.10060423612594604, 0.11577785015106201, 0.3959497809410095, -0.23205921053886414, -0.007092039100825787, 0.1421680897474289, -0.033782344311475754, 0.15470725297927856, -0.05985529348254204, -0.11284781247377396, 0.41878288984298706, 0.012480814009904861, 0.09610676020383835, 0.09385758638381958, 0.1206778883934021, -0.05080771818757057, -0.5067381262779236, -0.14283324778079987, -0.01962202973663807, -0.10288462787866592, 0.02445816993713379],
#        [0.3107169568538666, -0.16336588561534882, 0.13979989290237427, -0.06237269937992096, 0.20755282044410706, -0.18239142000675201, 0.2927098274230957, 0.17870883643627167, 0.15436840057373047, -0.20919054746627808, -0.014542280696332455, 0.08693742007017136, -0.039950814098119736, 0.14380718767642975, 0.2939481735229492, -0.007748228497803211, -0.04357427731156349, -0.07980721443891525, 0.08186690509319305, -0.03721674904227257, 0.6561188101768494, 0.1590929478406906, -0.06476886570453644, -0.03910057991743088, 0.025055011734366417]
#])

# After GD
coeffs_slag = jnp.asarray(
        [[0.007078533565397007, 0.04414522469913452, -0.23102717515184373, 0.41058933287870314, -0.169577483065633, 0.16970782362282869, 0.003616269913724381, 0.20086033934034336, -0.22773336937236702, 0.04531097780269283, 0.00641922385005893, 0.4397390574312386, 0.21960851832665984, 0.01876587527684584, 0.14781103733299195, -0.0028219774223087997, -0.15668019175326253, -0.15329711517903635, 0.4466588752002253, 0.035780654057182865, -0.24955543101196076, 0.016869715907615355, 0.03431255558519013, 0.22145653259809941, -0.0023747107995217777],
        [0.39351293088948336, 0.1589018810002318, 0.01081769736474947, 0.03606323072606645, -0.1127660627202977, 0.10241268994295735, 0.3978265615120608, -0.21382915670138467, 0.0069237983327537925, 0.16995437044654857, -0.01151149153829505, 0.16390568673368383, -0.07025475420668742, -0.17331503181436386, 0.4334855180412318, -0.0009567321868253532, 0.15285525988553592, 0.1495539739499433, 0.16525092959965315, 0.008833696590700883, -0.45217118218791014, -0.17266985323828543, 0.007535912291170834, -0.07155712209232318, -0.008779349532781177],
        [0.2790115996832558, -0.21159400988713775, 0.1296652500737074, -0.01674537313215033, 0.22485232092005528, -0.21783037045804576, 0.2656834984878348, 0.11469943967196074, 0.12722220369567278, -0.21974010873693373, 0.0022904998321640977, 0.06322014566973709, -0.061296215629195926, 0.1842931242702063, 0.2846574736643637, 0.012269498746779905, -0.07642688615494606, -0.08717308245189592, 0.05627546991734017, -0.0198188825434132, 0.6585888831453401, 0.18848296399959508, -0.017281814405495678, -0.07079227594585795, 0.004048466180244929]]         
)

# GD, but Lagrangian condition only
coeffs_slag = jnp.asarray(
[[-0.1358108388576733, 0.035766550545863855, -0.03807092614188383, 0.5675921401936409, -0.0516160975233029, 0.04840420536872968, -0.13333899390550197, 0.0003555930039491121, -0.04410224806095801, 0.03269275049347586, -0.02322092946233022, 0.5474577291237686, 0.03478475788636963, -0.025432467810294034, 0.0479938414080948, -0.009151172859612488, -0.011161079169012019, -0.0028551551251042528, 0.5462947698310406, 0.01805792841427549, -0.16094580399400743, -0.02900377727864876, 0.023721938776067528, 0.026611168166397275, -0.020776757829204087],
[0.4174914212085932, 0.011750411095709854, -0.01225243958683061, -0.03707815416197036, -0.013053554585259006, 0.016205554463864902, 0.419016827614462, -0.002858417985997926, -0.01120759677763754, 0.017921754265987175, 0.024147170643158297, 0.11682770957855419, 0.0019228277756606713, -0.019975331274587664, 0.4269277032276998, 0.018602229917295203, 0.004421830681035188, 0.0005978259451106599, 0.11707434760884373, 0.09325178127434021, -0.6531241745338725, -0.012874948539409495, 0.1064679808921474, 0.007801801803158518, 0.022440749029829996],
[0.3099179182826169, -0.04188824095560178, 0.041472174853552236, -0.08236731522052114, 0.04058188870531673, -0.05096657766880051, 0.30994670136069935, 0.003031945002826456, 0.03799566967549645, -0.044708650613846566, -0.014719673385737977, 0.028146874633559055, -0.032694185550436725, 0.029015105322894502, 0.2976107984625371, -0.01737256859438614, -0.0006160349605566706, -0.0007540481361117733, 0.026911760585267137, -0.11641687518018631, 0.8248395777049248, 0.03102532897245539, -0.1346091050957977, -0.0197067636767178, -0.015596274328359286]]
)

# GD, but Lagrangian condition only, 5000 epochs
coeffs_lag = jnp.asarray([[-0.1580693040293132, 0.0016182404585921408, 0.0028286872061590762, 0.5749822704183946, 0.0009041125503625326, -0.00021752003885359722, -0.1447961881076301, 4.8265599893633845e-05, -0.002076368843095339, -0.004303953819004004, -0.017280598615902185, 0.5479944564551311, 0.0004717765301682842, 0.002114441947826722, 0.04632174744030983, -0.007203748442471513, -0.0005527897316882424, 0.00014859382707993512, 0.5515512160146624, 0.019148038858791735, -0.1266817918866054, 0.0006184480383272959, 0.019721381814970525, -0.0005468211026641777, -0.017310518625623814],
[0.41327397522940773, 0.00018620714645262856, 0.001632481604305648, -0.03741685032558684, 0.0031251124146415674, 0.0020097602445332966, 0.41652381913536324, 0.00035268235122406095, 0.000388745742146067, 0.00047582630887633374, 0.024821350539927043, 0.12785551384894658, -0.0025905435628787066, -0.001475019239420548, 0.4263904030274426, 0.019212312467950556, 0.00017615748037153625, -0.0009351612617398123, 0.1195408627348049, 0.1069812606092351, -0.655402478754062, 0.0019890236282819416, 0.10764118899055074, 0.0015617796809589092, 0.023352299566278028],
[0.33753745815806174, 0.0017157855485682707, 9.495006408908772e-05, -0.11249187223454975, -0.002321407595917999, -7.344192087437665e-05, 0.33856005144333867, -0.00041809980785964676, 0.0005948639933105428, 0.002782728316608129, -0.016164304503578423, 0.01699181613435511, 0.0010317987085470344, -0.00124651729340921, 0.3175023267009879, -0.020356413756314293, -0.0022153745438325883, -0.0015469273989753448, 0.009980985654083082, -0.13000399165241008, 0.8000377898534957, 0.00046411274869782865, -0.13100580620868202, 0.0016296954023018902, -0.01646670760654718]])

# GD, but special condition only
coeffs_slag = jnp.asarray(
[[-0.07821336754598626, 0.05486480445697324, -0.21128696275983852, 0.44285304158042743, -0.14209252830801516, 0.1391058660019118, -0.07918747385310974, 0.21496134769662303, -0.1661424649999295, 0.05645742598397549, 0.0765739650035044, 0.4440448028931023, 0.17826262078379332, -0.010052414062434385, 0.06579865495012466, 0.07047014518912977, -0.12978716169756757, -0.12780038386128176, 0.46969200163334063, -0.1304256198318693, -0.25258717269418196, -0.005248031091144885, -0.1598742717324744, 0.21692931406039598, 0.07131124824967466],
[0.34090195628767384, 0.13451248413090522, 0.05437085966750525, 0.07566469574842542, -0.051734346705768146, 0.07759231890153856, 0.33617517324387003, -0.3541591141480331, 0.059140685030013684, 0.18053861444876307, 0.0329323076629561, 0.19450609802380206, -0.11700927777020133, -0.18093079161006562, 0.38332147744890877, 0.01933015135411308, 0.1976103779155955, 0.16121777496254436, 0.2401401778046355, -0.09756638675411913, -0.4059485276923816, -0.17026983993576372, -0.11882622330016604, -0.1152818431706068, 0.022439523149055424],
[0.21405486024989, -0.21654347913485417, 0.10690651924351167, 0.09419831948009257, 0.22559368496032814, -0.20704150989070438, 0.2061753093936398, 0.0844798938822647, 0.12638048709294047, -0.2335384369787678, -0.002261442640135554, 0.13343694414862053, -0.0787363269277066, 0.19576254801251936, 0.24638281269177129, 0.03019431380521164, -0.06608194623622618, -0.06568987099484624, 0.07627169804899739, -0.05628872476960828, 0.7041374923318728, 0.17123879344638926, -0.0869174982488026, -0.06753712375006206, 0.007195758825297712]]
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
#coeffs = coeffs_lag

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
print("Min set: ", min_set[:50])
print(f"min_set_distance: Min: {jnp.min(distances)}, Max: {jnp.max(distances)}, Mean: {jnp.mean(distances)}")

#with open("min_set_psi0.pkl", "wb") as f:
#    pickle.dump(min_set, f)

frobenius_norms = jnp.linalg.norm(kahler_form_restricted_normalized, axis=(1, 2))
# Pick the smallest 90% to avoid numerical issues
sorted_norms = jnp.sort(frobenius_norms)
norms_cut = sorted_norms[:int(sorted_norms.shape[0]*0.95)]
print(f"Kahler loss: Min: {jnp.min(norms_cut)}, Max: {jnp.max(norms_cut)}, Mean: {jnp.mean(norms_cut)}")

values, indices = jax.lax.top_k(frobenius_norms, 10)
#print("Largest 20 values:", values)
#print("Indices of the largest 20 values:", indices)

#for i in range(50):
#print('Point:', min_set[indices])
#print('kahler_form_restricted: ', kahler_form_restricted_normalized[indices])
#print('kahler_form_restricted norm: ', jnp.linalg.norm(kahler_form_restricted_normalized[indices], axis=(1,2)))
#print('top k norm values', values)
#print('restriction:', restriction[indices])
print('phases: ', phases)
print('phases mean: ', jnp.mean(phases))
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
make_fitness_plots(points_real, coeffs, psi, k=10000, n_refine_steps=50, metric=metric, compare_with_random=True, parent_folder='slag_psi0_GD_lag')

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
#print('First 1000 norms: ', norms_cut[:1000])
#Jprint('First 5000 norms: ', norms_cut[:5000])

#distances_sorted = jnp.sort(distances)
#print('First 5000 distances: ', distances_sorted[:5000])
#print('Last 20 distances: ', distances_sorted[-20:])

#PATCH_ACTIVE_INDICES = jnp.array([
#    [1, 2, 3, 4, 6, 7, 8, 9],  # patch=0: skip 0,6
#    [0, 2, 3, 4, 5, 7, 8, 9],  # patch=1: skip 1,6
#    [0, 1, 3, 4, 5, 6, 8, 9],  # patch=2: skip 2,6
#    [0, 1, 2, 4, 5, 6, 7, 9],  # patch=3: skip 3,6
#    [0, 1, 2, 3, 5, 6, 7, 8],  # patch=4: skip 4,6
#], dtype=jnp.int32)
#
#from get_restriction import compute_affine_jacobian
#from helper import (evaluate_equations_single_point, convert_complex_to_real_single,
#                    convert_real_to_complex_single, determine_patch_and_rescale_single)
#
#test_point = min_set_real[300]
#p_complex = convert_real_to_complex_single(test_point)
#p_complex_rescaled, patch_index = determine_patch_and_rescale_single(p_complex)
#
#active_indices = PATCH_ACTIVE_INDICES[patch_index]

#f_vec = evaluate_equations_single_point(test_point, coeffs, psi)
#J = compute_affine_jacobian(test_point, patch_index, coeffs, psi)

#print('f_vec: ', f_vec)
#print('J: ', J)
#JJT = J @ J.T + 1e-8 * jnp.eye(J.shape[0])
#w = jnp.linalg.solve(JJT, -f_vec)
#delta_p_active = J.T @ w

#print('JJT: ', JJT)
#print('w: ', w)
#print('delta_p_active: ', delta_p_active)
#
#test_point_new = test_point.at[active_indices].add(0.1*delta_p_active)
#
#res = evaluate_equations_single_point(test_point, coeffs, psi)
#res_new = evaluate_equations_single_point(test_point_new, coeffs, psi)
#
#test_point_new = convert_real_to_complex_single(test_point_new)
#test_point = convert_real_to_complex_single(test_point)
#
#test_point_new_rescaled, patch_index = determine_patch_and_rescale_single(test_point_new)

#print('before: ', test_point)
#print('after: ', test_point_new)
#print('after scaled: ', test_point_new_rescaled)
#print('res_old: ', res)
#print('res_new: ', res_new)
#
#test_point_new_rescaled_real = convert_complex_to_real_single(test_point_new_rescaled)
#res_new_rescaled = evaluate_equations_single_point(test_point_new_rescaled_real, coeffs, psi)
#print('res_new_rescaled: ', res_new_rescaled)
#
