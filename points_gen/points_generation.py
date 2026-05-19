"""Generate a point cloud on the (possibly deformed) Fermat quintic.

Outputs a pickle covering all 5 affine patches — the canonical input consumed
by GA.py / gradient_descent.py / etc.

Filename convention (matches what the rest of the pipeline expects):

    {out_dir}/1mil_patch_all_psi{int(psi.real)}_seed{seed}.pkl

Example
-------
    python points_generation.py --psi 0  --seed 1024 --out_dir ./data_psi/
    python points_generation.py --psi 10 --seed 1024 --out_dir ./data_psi/

Requires sympy + MLGeometry (JAX backend). See pyproject.toml [points-gen] extra.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import sympy as sp
import MLGeometry as mlg

# Allow running directly from points_gen/ — pick up helper.py from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from helper import dwork_filename


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--psi', type=complex, required=True,
                        help='Quintic deformation parameter (psi=0 is Fermat).')
    parser.add_argument('--seed', type=int, default=1024,
                        help='Random seed for NumPy (and any RNG MLGeometry consumes).')
    parser.add_argument('--n_pairs', type=int, default=200000,
                        help='Pairs per patch (final cloud has ~5 * n_pairs points).')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to write the output pickle into.')
    args = parser.parse_args()

    np.random.seed(args.seed)

    z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
    Z = [z0, z1, z2, z3, z4]
    f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + args.psi * z0 * z1 * z2 * z3 * z4

    print(f'Generating {args.n_pairs} pairs/patch, psi={args.psi}, seed={args.seed}')
    HS_train = mlg.hypersurface.Hypersurface(Z, f, args.n_pairs)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, dwork_filename(args.psi, seed=args.seed))
    with open(out_path, 'wb') as fp:
        pickle.dump(HS_train.points, fp)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
