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
newton_refine_steps = 60
psi = 1000000
#with open('/projects/ruehlehet/yidi/sLag/data/5mil_patch0_343.pkl', 'rb') as f:
with open(f'/projects/ruehlehet/yidi/sLag/data_psi/5mil_patch0_psi{psi}_seed1024.pkl', 'rb') as f:
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

coeffs_new2_old = jnp.asarray(
[[ 0.3930962,   0.        ,  0.        , -0.1971123 , -0.14913972, -0.01050573,
  -0.03096225, -0.24029773, -0.02394878,  0.04012342,  0.78441393,  0.06768256,
  -0.15905517, -0.062086  ,  0.15934001,  0.32446176,  0.00471163,  0.18783152,
   0.08087511,  0.2513211 , -0.15794371, -0.2771842 ,  0.06865574,  0.06018861,
   0.4955574 ],
 [ 0.        ,  0.3827779 ,  0.        ,  0.27206326, -0.16624802,  0.04201639,
  -0.1844598 , -0.12416575, -0.02687686, -0.0113407 ,  0.7416247 ,  0.24232002,
   0.26877552,  0.1858964 ,  0.03042048,  0.01189734,  0.3750462 ,  0.1868444,
  -0.11411023,  0.04271036,  0.03877318,  0.10707161,  0.27720955,  0.13165788,
  -0.00653032],
 [ 0.        ,  0.        ,  0.02324561, -0.11727799,  0.16256237, -0.03166607,
   0.4852623 ,  0.10793307, -0.04599528, -0.380682  ,  0.19350672, -0.08336116,
  -0.311158  , -0.15808216,  0.02345732, -0.02155375, -0.21765997, -0.267524,
   0.46060374, -0.05332949, -0.07048529, -0.11769748, -0.1551992 , -0.21934652,
  -0.09327249]])


coeffs_new2_wrong = jnp.asarray(
[[ 5.40679395e-02 , 0.00000000e+00 , 0.00000000e+00, -2.69871037e-02,
   1.13150209e-01 ,-3.99974018e-01 , 1.46702483e-01, -1.69987902e-01,
   4.73884754e-02 , 5.34256883e-02 ,-1.27399847e-01, -1.19332388e-01,
  -2.31782962e-02 , 8.87169316e-02 ,-1.85184613e-01,  1.37282848e-01,
  -8.51138309e-02 ,-4.26765114e-01 , 8.05996656e-02,  5.68166189e-03,
   4.65897232e-01 , 1.39971703e-01 ,-6.62859529e-02,  4.90132958e-01,
   1.92892000e-01],                                                 
 [ 0.00000000e+00 , 1.96709201e-01 , 0.00000000e+00, -7.32468767e-03,
  -3.51928443e-01 , 1.09845646e-01 ,-2.63207871e-02,  1.76860020e-01,
  -6.96608871e-02 , 4.94183786e-03 ,-7.88269401e-01,  2.40838025e-02,
  -3.02682579e-01 ,-2.24210992e-02 , 1.26684070e-01, -1.15644634e-01,
  -5.50659448e-02 , 1.24451518e-01 ,-4.57903370e-02, -2.37067595e-01,
  -2.62883067e-01 ,-8.72500986e-02 ,-3.11325163e-01, -3.00658107e-01,
  -4.42873567e-01],                                                 
 [ 0.00000000e+00 , 0.00000000e+00 , 2.36074645e-02, -5.01734503e-02,
  -3.67213599e-02 , 2.04631582e-01 , 1.45315230e-01,  2.64527231e-01,
  -3.28038871e-01 ,-1.43583892e-02 , 5.52090642e-04,  5.94248399e-02,
  -8.19587186e-02 ,-2.93176249e-02 , 1.57117099e-01, -8.54549631e-02,
   1.47648692e-01 , 5.47227077e-02 ,-9.21246316e-03,  9.34689641e-02,
  -3.30991536e-01 ,-1.30555779e-01 ,-1.84107155e-01, -5.09584665e-01,
  -7.52168059e-01]])

coeffs_new2 = jnp.asarray(
[[ 0.07606348 , 0.        ,  0.        , -0.5654261 ,  0.391762  ,  0.18431653,
   0.01693727 , 0.09512562,  0.10052733, -0.16347681, -0.13251261, -0.34130803,
   0.21124752 ,-0.15984906,  0.16258587, -0.22203413, -0.04050797, -0.2726697 ,
  -0.17400618 , 0.01005866, -0.03200786, -0.2509263 , -0.1252478 , -0.07330906,
   0.14720553],
 [ 0.         , 0.14121912,  0.        ,  0.5801074 , -0.0718888 , -0.18661004,
  -0.01104976 ,-0.11662827, -0.0964903 ,  0.560512  ,  0.04962098,  0.23772691,
  -0.03980598 ,-0.20979501, -0.05179332, -0.01875344, -0.3097298 ,  0.03023832,
   0.15302064 ,-0.01899938,  0.00725389, -0.12094607,  0.03468487,  0.06942332,
   0.19792205],
 [ 0.         , 0.        ,  0.1960515 ,  0.40224233, -0.0455599 , -0.10191692,
   0.1492465  ,-0.09816858,  0.04283687, -0.19708976,  0.8123748 , -0.30850843,
   0.24762854 ,-0.08567714,  0.01246204,  0.49490038, -0.021838  , -0.16351257,
   0.04008869 , 0.10562648,  0.03963509,  0.05808787,  0.11497162, -0.06025742,
   0.31207192]]
)

coeffs_1e6 = jnp.asarray(
[[ 0.16887893 , 0.        ,  0.        ,  0.12236486, -0.38354298, -0.02861437,
   0.02112332 ,-0.04184642, -0.12716532, -0.1085003 , -0.72523546,  0.01922101,
  -0.36247265 ,-0.02121013, -0.05375981, -0.30833504, -0.03466301,  0.01922832,
  -0.30355275 , 0.22291242,  0.03664982,  0.20086268, -0.2948762 , -0.09085743,
   0.49162424],                                                               
 [ 0.         , 0.16347654,  0.        , -0.01169971, -0.06434734, -0.0673325 ,
  -0.06184864 , 0.01372362,  0.16961709,  0.21968496,  0.8737739 ,  0.02701011,
   0.03073046 ,-0.00308765,  0.00487616,  0.24270375, -0.11748812, -0.2239338 ,
   0.0624653  , 0.7053649 ,  0.08507178,  0.09725375, -0.12915121,  0.2922963 ,
   0.2947036 ],                                                               
 [ 0.         , 0.        ,  0.08588855,  0.01362793,  0.05483339,  0.09738956,
   0.0272746  , 0.10507316,  0.05883223,  0.04458414,  0.93061185, -0.02784354,
   0.06543607 , 0.02157319, -0.04990697, -0.23928161,  0.0111317 , -0.16446757,
   0.25231323 , 0.03661969,  0.00168569,  0.09530029,  0.6152934 ,  0.0738647 ,
   0.6288285 ]]
)

# Psi = 1e6, gen417 which has a total fitness ~0.97
coeffs_1e6 = jnp.asarray(
[[ 4.4302639e-02 , 0.0000000e+00,  0.0000000e+00 , 1.7205824e-01,
  -3.7917268e-01 ,-8.7848091e-03,  1.9989641e-02 , 5.8316067e-02,
  -1.6170441e-01 ,-1.0400223e-01, -9.4998211e-01 ,-2.1822026e-03,
  -1.4249712e-01 ,-3.2513195e-03, -4.6802247e-03 ,-5.7566877e-02,
  -1.2484960e-02 , 2.0855201e-02, -2.8606266e-01 , 2.1250857e-01,
   9.6409835e-02 , 1.9763897e-01, -1.5274563e-01 ,-1.0357033e-01,
   5.1929832e-01],                                               
 [ 0.0000000e+00 , 1.2052754e-01,  0.0000000e+00 , 6.0000233e-02,
  -5.0960928e-02 , 5.1575821e-02, -7.9929203e-02 , 6.0945589e-02,
   7.7849716e-02 , 4.8065439e-02,  8.7490487e-01 ,-7.5175492e-03,
  -2.3426437e-04 , 3.6833391e-03, -1.4522955e-02 , 4.7582403e-02,
  -3.6841422e-02 ,-1.2648566e-01,  7.3470578e-02 , 7.7087039e-01,
   1.2338873e-01 , 1.5169685e-01,  1.4306273e-01 , 2.8308514e-01,
   5.0103498e-01],                                               
 [ 0.0000000e+00 , 0.0000000e+00,  6.4261765e-03 , 1.2481595e-02,
  -1.8729988e-01 , 1.5637019e-01, -1.9581767e-02 , 2.0940706e-01,
  -9.3427442e-02 ,-1.7067821e-01,  4.8100442e-02 ,-3.7274882e-02,
  -2.6532167e-01 , 5.0166454e-02,  3.3137098e-01 ,-2.7362883e-01,
  -1.0506800e-01 ,-1.4207754e-01, -1.5860933e-01 , 4.9244344e-01,
  -1.1220082e-01 ,-7.2976716e-02,  7.9400891e-01 , 3.7674136e-02,
   4.9655265e-01]]
)

coeffs_1e6 = jnp.asarray(
[[ 0.01882634 , 0.        ,  0.        ,  0.20086911, -0.28655097,  0.06633254,
   0.1113897  ,-0.27126813,  0.19795269,  0.15024045,  0.4959268 ,  0.15134832,
   0.01135873 ,-0.11218777,  0.19362706, -0.02116979,  0.1201715 ,  0.4512422 ,
  -0.06058701 ,-0.09518168,  0.40854874,  0.33558062,  0.07065602, -0.13077357,
   0.08173177],                                                               
 [ 0.         , 0.06365854,  0.        ,  0.09473755, -0.08287388,  0.10629219,
   0.18214862 ,-0.09060092,  0.10901298, -0.19384849,  0.79580015, -0.0290449 ,
   0.10657115 , 0.09566469,  0.04620985,  0.01500307,  0.17323062, -0.12566367,
   0.00446166 , 0.90239257, -0.05087172, -0.03285413,  0.319952  ,  0.08289158,
   0.21452284],                                                               
 [ 0.         , 0.        ,  0.03114937, -0.00793634, -0.1766626 , -0.05271116,
   0.53059006 , 0.02424736,  0.06547625,  0.02567908, -0.5115609 , -0.13918906,
   0.04664252 ,-0.10235695, -0.01453263,  0.01552396,  0.01616347, -0.2802434 ,
   0.07653465 , 0.31979272,  0.5092687 , -0.08959255, -0.40888858, -0.15703064,
  -0.08111086]]
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

# Psi = 1e6, search with lagrangian > 0.98 then added special
# fitness = 1.90147
coeffs_1e6_add_cond = jnp.asarray(
[[ 9.53661976e-04 , 0.00000000e+00 , 0.00000000e+00,  9.04977173e-02,
   1.27847672e-01 , 1.53666481e-01 , 3.08604296e-02, -3.89056616e-02,
  -3.85315627e-01 ,-1.85766276e-02 ,-9.70079839e-01, -4.62175459e-02,
  -4.29277457e-02 ,-1.69853672e-01 ,-5.00739627e-02, -3.28845195e-02,
   1.54963821e-01 , 3.50431681e-01 , 1.24888726e-01, -2.96277434e-01,
   1.64248616e-01 ,-4.72024679e-02 ,-9.14792940e-02, -5.44964448e-02,
  -3.20947796e-01],                                                 
 [ 0.00000000e+00 , 5.37098013e-02 , 0.00000000e+00,  7.36275762e-02,
  -1.09184340e-01 , 1.04280382e-01 ,-8.90227258e-02,  3.34747620e-02,
   2.19741706e-02 , 7.10350052e-02 ,-8.63074183e-01,  7.59102078e-03,
  -4.45095487e-02 ,-3.34559567e-02 ,-1.48880020e-01, -1.57444030e-02,
   5.45533970e-02 , 1.28053516e-01 , 4.49026600e-02, -4.31205571e-01,
   9.84440222e-02 ,-2.03096926e-01 ,-5.85947394e-01,  1.62132233e-01,
  -6.21571898e-01],                                                 
 [ 0.00000000e+00 , 0.00000000e+00 , 3.73326182e-01,  7.50759840e-02,
  -1.17065467e-01 ,-8.90509225e-03 , 1.98842343e-02,  4.25748713e-03,
  -7.27512613e-02 , 3.80871385e-01 ,-7.33465910e-01, -1.03102028e-02,
  -2.79833108e-01 ,-1.07549867e-02 ,-4.02453393e-02, -1.94502398e-02,
  -1.26002789e-01 , 2.43270159e-01 ,-4.95188385e-02, -6.77733421e-01,
  -7.08549842e-02 ,-6.67372420e-02 ,-1.96968898e-01,  3.75721492e-02,
  -8.55571553e-02]]
)
perturbation_order = 0.001

seed = 1230
#seed = 42
key = jax.random.PRNGKey(seed)
coeffs_random = jax.random.uniform(key, (3, 25), minval=-1, maxval=1)

#coeffs = coeffs_new2
#coeffs = coeffs_new
#coeffs = coeffs_random
#coeffs = coeffs_RP3 + perturbation_order * coeffs_random
#coeffs = coeffs_T3
#coeffs = coeffs_T3 + perturbation_order * coeffs_random
coeffs = coeffs_1e6
#coeffs = coeffs_1e6_add_cond

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
min_set_real, distances = filter_and_refine(points_real, coeffs, psi, k=10000, n_refine_steps=newton_refine_steps, constant_coord=0, debug_mode=True)
#min_set_real = filter_and_refine(points_real, coeffs, jacobian_func, psi, k=3000, n_refine_steps=5, constant_coord=0)
total_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted_normalized, phases = compute_combined_fitness(min_set_real, coeffs, psi, debug_mode=True)
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

frobenius_norms = jnp.linalg.norm(kahler_form_restricted, axis=(1, 2))
# Pick the smallest 90% to avoid numerical issues
sorted_norms = jnp.sort(frobenius_norms)
norms_cut = sorted_norms[:int(sorted_norms.shape[0]*1.0)]
print(f"Kahler loss: Min: {jnp.min(norms_cut)}, Max: {jnp.max(norms_cut)}, Mean: {jnp.mean(norms_cut)}")

values, indices = jax.lax.top_k(norms_cut, 20)
#print("Largest 20 values:", values)
#print("Indices of the largest 20 values:", indices)

#for i in range(50):
print('Point:', min_set[indices])
print('kahler_form_restricted: ', kahler_form_restricted_normalized[indices])


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
print(counts)
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
