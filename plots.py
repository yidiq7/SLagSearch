import jax
import jax.numpy as jnp
import numpy as np
import os
import re
import pickle
import matplotlib.pyplot as plt
from functools import partial
from find_smooth_submanifold import filter_and_refine, normalize_coeffs
from slag_condition import (
    compute_holomorphic_form,
    compute_kahler_form_restricted,
    compute_kahler_form_unrestricted,
    compute_special_condition_fitness,
    vmap_compute_affine_jacobian,
    vmap_compute_restriction,
)
from get_restriction import compute_Omega_restriction
from helper import canonicalize_coeffs, convert_real_to_complex_batch, determine_patches_batch
from typing import Optional


@partial(jax.jit, static_argnames=('metric',))
def _per_chunk_diagnostics(
    min_set_real_chunk: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    metric: str,
):
    """Returns per-point (frobenius_norms_plot, norms_for_fitness, phases_2pi).

    Phases are returned on [0, 2*pi) for plotting (so the user can see the
    raw distribution, not the mod-pi reduction used by training).
    frobenius_norms_plot mirrors compute_combined_fitness debug-mode output:
      ||K_restricted||_F / sqrt(||K_unrestricted||_F)
    norms_for_fitness is what compute_lagrangian_condition_fitness uses:
      ||K_restricted||_F / ||K_unrestricted||_F
    """
    min_set = convert_real_to_complex_batch(min_set_real_chunk)
    patch_indices = determine_patches_batch(min_set)
    jacobians = vmap_compute_affine_jacobian(min_set_real_chunk, patch_indices, coeffs, psi)
    restrictions = vmap_compute_restriction(jacobians)
    kahler_unrestricted = compute_kahler_form_unrestricted(min_set, patch_indices, metric=metric)
    kahler_restricted = compute_kahler_form_restricted(min_set, restrictions, patch_indices, metric=metric)
    norms_unrestricted = jnp.linalg.norm(kahler_unrestricted, axis=(1, 2))
    norms_restricted = jnp.linalg.norm(kahler_restricted, axis=(1, 2))
    frobenius_norms_plot = norms_restricted / jnp.sqrt(norms_unrestricted)
    norms_for_fitness = norms_restricted / norms_unrestricted
    # Bypass compute_holomorphic_form_restricted so we can use mod 2*pi for the
    # plot; training/fitness still use the mod-pi reduction (applied below).
    Omega, _, Omega_coord = compute_holomorphic_form(min_set, patch_indices, psi)
    Omega_restriction = compute_Omega_restriction(restrictions, Omega_coord)
    phases_2pi = jnp.angle(Omega * Omega_restriction) % (2 * jnp.pi)
    return frobenius_norms_plot, norms_for_fitness, phases_2pi


def _chunked_diagnostics(
    min_set_real: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    metric: str,
    chunk_size: int,
):
    """Loops _per_chunk_diagnostics over points, transferring each chunk to
    host memory as numpy so GPU memory only ever holds one chunk's intermediates.
    """
    N = min_set_real.shape[0]
    fn_chunks, ff_chunks, ph_chunks = [], [], []
    for i in range(0, N, chunk_size):
        fn, ff, ph = _per_chunk_diagnostics(
            min_set_real[i:i + chunk_size], coeffs, psi, metric
        )
        fn_chunks.append(np.asarray(fn))
        ff_chunks.append(np.asarray(ff))
        ph_chunks.append(np.asarray(ph))
    return (
        np.concatenate(fn_chunks),
        np.concatenate(ff_chunks),
        np.concatenate(ph_chunks),
    )

def make_fitness_plots(
    points_real: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    k: int = 100000,
    n_refine_steps: int = 100,
    metric: str = 'FS',
    compare_with=None,
    parent_folder: Optional[str] = 'plots_slag',
    patch_index: Optional[int] = None,
    chunk_size: int = 10000,
    primary_label: str = 'Potential sLag',
    compare_label: str = 'Random intersection',
    primary_color: str = 'skyblue',
    compare_color: str = 'orange',
    fix_kahler_x_range: bool = True,
    ) -> None:
    """Plot Kahler-norm and Omega-phase distributions for `coeffs`, optionally
    overlaid with a comparison distribution.

    compare_with:
        None         -- primary only
        "random"     -- generate random coeffs (matching `coeffs.shape`) and overlay
        jnp.ndarray  -- use these coeffs as the comparison set

    primary_color / compare_color / primary_label / compare_label tune the
    histograms; fix_kahler_x_range=True pins both the bin range and xlim to [0, 3].
    """
    os.makedirs(parent_folder, exist_ok=True)

    # --- Primary set ---
    min_set_real, distances, _ = filter_and_refine(
        points_real, coeffs, psi, k, n_refine_steps
    )
    if patch_index:
        patch_indices = determine_patches_batch(convert_real_to_complex_batch(min_set_real))
        idx = jnp.where(patch_indices == patch_index)
        min_set_real = min_set_real[idx]
        distances = distances[idx]

    frobenius_norms, norms_for_fitness, phases = _chunked_diagnostics(
        min_set_real, coeffs, psi, metric, chunk_size
    )
    sorted_nf = np.sort(norms_for_fitness)
    cutoff = int(sorted_nf.shape[0] * 0.99)
    lagrangian_fitness = float(np.exp(-10.0 * np.mean(sorted_nf[:cutoff])))
    # special_fitness uses mod-pi phases (consistent with training);
    # the histograms below use the raw mod-2pi phases.
    phases_mod_pi = phases % np.pi
    special_fitness = float(compute_special_condition_fitness(jnp.asarray(phases_mod_pi), n_bins=100))

    print(f"min_set_distance: Min: {jnp.min(distances)}, Max: {jnp.max(distances)}, Mean: {jnp.mean(distances)}")
    print(f"Lagrangian fitness: {lagrangian_fitness}, special_fitness: {special_fitness}")

    # --- Comparison set ---
    has_compare = compare_with is not None
    if has_compare:
        if isinstance(compare_with, str):
            if compare_with != "random":
                raise ValueError(f"compare_with={compare_with!r}; expected None, 'random', or an ndarray.")
            key = jax.random.PRNGKey(1230)
            cmp_coeffs = jax.random.uniform(key, coeffs.shape, minval=-1, maxval=1)
            cmp_coeffs = canonicalize_coeffs(cmp_coeffs)
            cmp_coeffs = normalize_coeffs(cmp_coeffs)
        else:
            cmp_coeffs = compare_with
        min_set_real_cmp, _, _ = filter_and_refine(
            points_real, cmp_coeffs, psi, k, n_refine_steps
        )
        frobenius_norms_cmp, _, phases_cmp = _chunked_diagnostics(
            min_set_real_cmp, cmp_coeffs, psi, metric, chunk_size
        )

    # --- Kahler-form histogram ---
    plt.figure(figsize=(10, 6))
    hist_kwargs = dict(bins=200, alpha=0.7, density=True)
    if fix_kahler_x_range:
        hist_kwargs['range'] = (0, 3)
    plt.hist(frobenius_norms, label=primary_label, color=primary_color, **hist_kwargs)
    if has_compare:
        plt.hist(frobenius_norms_cmp, label=compare_label, color=compare_color, **hist_kwargs)
    if fix_kahler_x_range:
        plt.xlim(0, 3)
    plt.xlabel('Frobenius norm')
    plt.ylabel('Probability density')
    plt.title('Distribution of the norm of the Kahler form')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(parent_folder, 'Kahler_form_loss_histogram.png'))
    plt.close()

    # --- Phase histogram (polar, always [0, 2*pi)) ---
    number_of_bins = 1000
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    width = 2 * np.pi / number_of_bins
    counts, bin_edges = np.histogram(phases, bins=number_of_bins, range=(0, 2 * np.pi))
    angles = bin_edges[:-1]
    max_count = int(counts.max())
    if has_compare:
        counts_cmp, _ = np.histogram(phases_cmp, bins=number_of_bins, range=(0, 2 * np.pi))
        max_count = max(max_count, int(counts_cmp.max()))
    baseline_radius = max_count / 2

    ax.bar(angles, counts, width=width, alpha=0.7, color=primary_color,
           label=primary_label, bottom=baseline_radius)
    if has_compare:
        ax.bar(angles, counts_cmp, width=width, alpha=0.7, color=compare_color,
               label=compare_label, bottom=baseline_radius)

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2'], fontsize=12)
    if has_compare:
        radial_grid_values = [baseline_radius + max_count * 0.25,
                              baseline_radius + max_count * 0.5,
                              baseline_radius + max_count * 0.75]
    else:
        radial_grid_values = [baseline_radius, baseline_radius + max_count * 0.5]
    ax.set_rgrids(radial_grid_values, angle=22.5)
    ax.set_yticklabels([])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_rlim(0, baseline_radius + max_count * 1.05)
    ax.set_title('Distribution of the phases of the holomorphic 3-form', fontsize=16, pad=25)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    plt.savefig(os.path.join(parent_folder, 'circular_phase_histogram.png'), bbox_inches='tight')
    plt.close()

    make_scatter_plots(min_set_real, frobenius_norms, parent_folder)
    save_min_set(min_set_real, parent_folder)


def make_scatter_plots(
    min_set_real: jnp.ndarray,
    frobenius_norms: jnp.ndarray,
    parent_folder: str
):

    lagrangian_fitness = jnp.exp(-10*frobenius_norms)
    # Define the coordinate pairs and labels for plotting.
    plot_configs = [
        {'x_idx': 0, 'y_idx': 5, 'xlabel': 'z0 real', 'ylabel': 'z0 imag', 'file_part': 'z0rz0i'},
        {'x_idx': 1, 'y_idx': 2, 'xlabel': 'z1 real', 'ylabel': 'z2 real', 'file_part': 'z1rz2r'},
        {'x_idx': 3, 'y_idx': 4, 'xlabel': 'z3 real', 'ylabel': 'z4 real', 'file_part': 'z3rz4r'},
        {'x_idx': 6, 'y_idx': 7, 'xlabel': 'z1 imag', 'ylabel': 'z2 imag', 'file_part': 'z1iz2i'},
        {'x_idx': 8, 'y_idx': 9, 'xlabel': 'z3 imag', 'ylabel': 'z4 imag', 'file_part': 'z3iz4i'},
        {'x_idx': 1, 'y_idx': 6, 'xlabel': 'z1 real', 'ylabel': 'z1 imag', 'file_part': 'z1rz1i'},
    ]

    for config in plot_configs:
        min_set_x = min_set_real[:, config['x_idx']]
        min_set_y = min_set_real[:, config['y_idx']]

        # --- Colored by Lagrangian Fitness ---
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(min_set_x, min_set_y, c=lagrangian_fitness, cmap='viridis', s=0.05, edgecolor=None)
        plt.colorbar(scatter, label='Lagrangian Fitness')
        plt.title(f'Scatter Plot of {config["ylabel"]} vs {config["xlabel"]} (Color by Lagrangian Fitness)')
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        plt.grid(True, linestyle='--', alpha=0.6)
        output_filename = os.path.join(parent_folder, f"scatter_{config['file_part']}.png")
        plt.savefig(output_filename, dpi=300)
        plt.close()

def save_min_set(min_set_real: jnp.ndarray, parent_folder: str) -> None:
    min_set = min_set_real[:,:5]+min_set_real[:,5:]*1j
    with open(os.path.join(parent_folder, "min_set.pkl"), "wb") as f:
        pickle.dump(min_set, f) 


def plot_slag_data(jobid, max_rank, coordinates):
    """
    Finds, loads, and plots data from GA output folders, automatically
    creating a 2D or 3D scatter plot based on the length of 'coordinates'.

    Args:
        jobid (str or int): The job ID used in the folder names.
        max_rank (int): The maximum rank to include in the plot.
        coordinates (tuple of int): A tuple of 2 or 3 integer indices 
                                    specifying which coordinates to plot.
    """
    # This list provides labels for the 10 real coordinates
    coord_list = [
        'z0_real', 'z1_real', 'z2_real', 'z3_real', 'z4_real',
        'z0_img', 'z1_img', 'z2_img', 'z3_img', 'z4_img'
    ]

    # --- 1. Argument Validation ---
    if not isinstance(coordinates, (list, tuple)) or not (2 <= len(coordinates) <= 3):
        print("Error: 'coordinates' must be a tuple or list containing 2 or 3 integers.")
        return

    main_folder = f"plots_slag_{jobid}"
    if not os.path.isdir(main_folder):
        print(f"Error: Base directory '{main_folder}' not found.")
        return

    # --- 2. Data Aggregation ---
    all_points = []
    pattern = re.compile(f"plots_slag_{jobid}_(\\d+)_id\\d+")
    print(f"Searching for subfolders in '{main_folder}' with rank < {max_rank}...")

    # Use sorted() to process folders in a predictable order
    for subfolder_name in sorted(os.listdir(main_folder)):
        full_path = os.path.join(main_folder, subfolder_name)
        if os.path.isdir(full_path):
            match = pattern.match(subfolder_name)
            if match:
                rank = int(match.group(1))
                if rank < max_rank:
                    pkl_path = os.path.join(full_path, "min_set.pkl")
                    if os.path.exists(pkl_path):
                        try:
                            with open(pkl_path, 'rb') as f:
                                min_set_complex = pickle.load(f)
                                min_set_complex_np = np.asarray(min_set_complex)
                                # Convert N x 5 complex to N x 10 real
                                min_set_real = np.hstack([min_set_complex_np.real, min_set_complex_np.imag])
                                all_points.append(min_set_real)
                        except Exception as e:
                            print(f"Warning: Could not process file {pkl_path}. Error: {e}")

    if not all_points:
        print("No data found to plot. Check folder names and paths.")
        return

    # Consolidate all data into a single NumPy array
    all_points_real = np.vstack(all_points)
    print(f"Plotting {len(all_points_real)} total points.")

    # --- 3. Plotting (2D or 3D based on input) ---
    fig = plt.figure(figsize=(13, 11))

    # --- 3A. 3D Plotting Logic ---
    if len(coordinates) == 3:
        c1, c2, c3 = coordinates
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            all_points_real[:, c1], 
            all_points_real[:, c2], 
            all_points_real[:, c3], 
            s=0.5, alpha=0.7
        )
        ax.set_title(f"3D Scatter Plot for Job ID: {jobid} (Ranks < {max_rank})", fontsize=16)
        ax.set_xlabel(f"{coord_list[c1]}", fontsize=12)
        ax.set_ylabel(f"{coord_list[c2]}", fontsize=12)
        ax.set_zlabel(f"{coord_list[c3]}", fontsize=12)
        filename = f'scatter_plot_3D_{jobid}_{c1}_{c2}_{c3}.png'

    # --- 3B. 2D Plotting Logic ---
    else: # This will be len(coordinates) == 2
        c1, c2 = coordinates
        ax = fig.add_subplot(111)
        ax.scatter(
            all_points_real[:, c1], 
            all_points_real[:, c2], 
            s=0.5, alpha=0.7, edgecolors='none'
        )
        ax.set_title(f"2D Scatter Plot for Job ID: {jobid} (Ranks < {max_rank})", fontsize=16)
        ax.set_xlabel(f"{coord_list[c1]}", fontsize=12)
        ax.set_ylabel(f"{coord_list[c2]}", fontsize=12)
        filename = f'scatter_plot_2D_{jobid}_{c1}_{c2}.png'
        
    # --- 4. Finalize and Save Plot ---
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    
    save_path = os.path.join(main_folder, filename)
    plt.savefig(save_path, dpi=150) # dpi improves resolution
    plt.close(fig) # Close the figure to free up memory
    print(f"Plot successfully saved to: {save_path}")

