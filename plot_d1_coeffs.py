"""Generate fitness plots for a fixed d=1 coefficient matrix (no GA).

Usage:
    python plot_d1_coeffs.py --parent_folder plots_d1_manual

Mirrors the post-GA plotting call in GA.py:761.
"""
import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from helper import assert_metric_psi_compatible, dwork_points_path, load_points
from plots import make_fitness_plots


jax.config.update('jax_default_matmul_precision', 'high')


PSI = 0+0j
METRIC = 'k4_fermat'


D1_COEFFS = jnp.asarray([
    [-0.7303538918495178, 0.08144077658653259, 0.0508277602493763, 0.1342628300189972, 0.10492400079965591, 0.026059845462441444, 0.08795911818742752, 0.18202504515647888, -0.2318553328514099, 0.19111207127571106, 0.12272824347019196, -0.18792606890201569, -0.010629810392856598, 0.03757120668888092, -0.03072214126586914, 0.1476466804742813, -0.09663095325231552, 0.013276210054755211, -0.02938280999660492, 0.0015832323115319014, -0.232550248503685, -0.24750050902366638, -0.09898191690444946, -0.25930696725845337, -0.13821059465408325],
    [-0.30552297830581665, 0.05322077125310898, 0.026228293776512146, 0.0475473515689373, 0.09884151816368103, -0.0022952028084546328, 0.07855669409036636, -0.5385099649429321, 0.48728707432746887, -0.5061385631561279, 0.05925785377621651, -0.08616615831851959, -0.05580515041947365, -0.03696516156196594, 0.04638112708926201, 0.06312626600265503, -0.04773535579442978, -0.018126241862773895, 0.013317487202584743, -0.03146995231509209, -0.10255347937345505, -0.15281900763511658, -0.008377314545214176, -0.19228313863277435, -0.07119646668434143],
    [-0.742755651473999, -0.02884053625166416, -0.09886687248945236, -0.027701135724782944, -0.19895869493484497, -0.16314652562141418, -0.2282542735338211, 0.1384347826242447, -0.12351837754249573, 0.09779400378465652, 0.08285680413246155, -0.07458371669054031, 0.22398249804973602, 0.21140094101428986, 0.19246235489845276, 0.10321154445409775, -0.10807567834854126, 0.029491063207387924, -0.023275744169950485, -0.009872819297015667, 0.17578807473182678, 0.17009639739990234, -0.06591708213090897, 0.14255480468273163, -0.14826683700084686],
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--points_file', default=None,
                        help='Override path to point cloud pkl. '
                             'Default: helper.dwork_points_path(psi).')
    parser.add_argument('--psi', type=complex, default=PSI)
    parser.add_argument('--metric', default=METRIC)
    parser.add_argument('--k', type=int, default=80000)
    parser.add_argument('--n_refine_steps', type=int, default=80)
    parser.add_argument('--parent_folder', default='plots_d1_manual')
    parser.add_argument('--compare_with', default='random',
                        help="'None', 'random', or unused (manual ndarray not exposed here).")
    args = parser.parse_args()

    assert_metric_psi_compatible(args.metric, args.psi)
    if args.points_file is None:
        args.points_file = dwork_points_path(args.psi, seed=1024)

    points_real = load_points(args.points_file)
    print(f"Loaded {points_real.shape[0]} points from {args.points_file}")

    compare_with = None if args.compare_with.lower() == 'none' else args.compare_with

    make_fitness_plots(
        points_real,
        D1_COEFFS,
        jnp.asarray(args.psi),
        k=args.k,
        n_refine_steps=args.n_refine_steps,
        metric=args.metric,
        compare_with=compare_with,
        parent_folder=args.parent_folder,
    )
    print(f"Plots written to {args.parent_folder}/")


if __name__ == '__main__':
    main()
