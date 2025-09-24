import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from find_smooth_submanifold import filter_and_refine, normalize_coeffs
from slag_condition import compute_combined_fitness
from helper import canonicalize_coeffs from typing import Optional

def make_fitness_plots(
    points_real: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    k: int = 100000,
    n_refine_steps: int = 100,
    constant_coord: int = 0,
    metric: str = 'FS',
    compare_with_random: bool = False,
    parent_folder: Optional[str] = 'plots_slag',
    ) -> None:

    # Create the folder for the plots 
    os.makedirs(parent_folder, exist_ok=True)

    # Compute the norms and phases 
    min_set_real, distances = filter_and_refine(points_real, coeffs, psi, k, n_refine_steps, constant_coord, debug_mode=True)
    total_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted, restriction, phases = compute_combined_fitness(min_set_real, coeffs, psi, constant_coord, metric, debug_mode=True)

    frobenius_norms = jnp.linalg.norm(kahler_form_restricted, axis=(1, 2))
    print(f"Lagrangian fitness: {lagrangian_fitness}, special_fitness: {special_fitness}")
    # Fitness plot
    if not compare_with_random:
        # Plot the Kahler form loss
        plt.figure(figsize=(10, 6))
        plt.hist(frobenius_norms, bins=200, alpha=0.7, label='Potential sLag', color='skyblue', density=True)
        plt.xlabel('Frobenius norm')
        plt.ylabel('Counts')
        plt.title('Distribution of the norm of the Kahler form')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(parent_folder, f'Kahler_form_loss_histogram.png'))
        plt.close()

        # Plot the phase of Omega
        number_of_bins = 1000
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

        # Define the width of each bar
        width = 2 * np.pi / number_of_bins
        counts, bin_edges = np.histogram(phases, bins=number_of_bins, range=(0, 2 * np.pi))
        angles = bin_edges[:-1]

        # --- Set baseline dynamically to half the max peak height ---
        max_count = counts.max()
        baseline_radius = max_count / 2

        ax.bar(angles, counts, width=width, alpha=0.7, color='skyblue', label='Potential sLag', bottom=baseline_radius)
        outer_limit = baseline_radius + max_count * 1.05

        # --- Format the plot ---
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)

        # --- Set angle labels to RADIANS ---
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2'], fontsize=12)

        # Set radial grid lines at 75% and 100% of the max height
        radial_grid_values = [baseline_radius, baseline_radius + max_count * 0.5]
        #radial_grid_values = [baseline_radius + max_count * 0.25, baseline_radius + max_count * 0.5, baseline_radius + max_count*0.75]
        ax.set_rgrids(radial_grid_values, angle=22.5)
        ax.set_yticklabels([]) # Hide the number labels on the grid
        ax.grid(True, linestyle='--', alpha=0.6) # Keep grid but make it faint

        # Adjust the plot's outer limit to fit the data perfectly
        ax.set_rlim(0, baseline_radius + max_count * 1.05)

        ax.set_title('Distribution of the phases of the holomorphic 3-form', fontsize=16, pad=25)
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        plt.savefig(os.path.join(parent_folder, f'circular_phase_histogram.png'), bbox_inches='tight')
        plt.close()

     # The option to plot both slag and random manifold in one plot for comparison
    elif compare_with_random:
        seed = 1230
        key = jax.random.PRNGKey(seed)
        coeffs_random = jax.random.uniform(key, (3, 25), minval=-1, maxval=1)
        coeffs_random =  canonicalize_coeffs(coeffs_random)
        coeffs_random =  normalize_coeffs(coeffs_random)

        min_set_real_random, distances_random = filter_and_refine(points_real, coeffs_random, psi, k, n_refine_steps, constant_coord, debug_mode=True)
        total_fitness_random, lagrangian_fitness_random, special_fitness_random, kahler_form_restricted_random, restriction_random, phases_random = compute_combined_fitness(min_set_real_random, coeffs_random, psi, constant_coord, metric, debug_mode=True)
        frobenius_norms_random = jnp.linalg.norm(kahler_form_restricted_random, axis=(1, 2))

        plt.figure(figsize=(10, 6))
        plt.hist(frobenius_norms, bins=200, alpha=0.7, label='Potential sLag', color='skyblue', density=True)
        plt.hist(frobenius_norms_random, bins=200, alpha=0.7, label='Random intersection', color='orange', density=True)
        plt.xlim(0, 1.5)
        plt.ylim(0, 300)
        plt.xlabel('Frobenius norm')
        plt.ylabel('Counts')
        plt.title('Distribution of the norm of the Kahler form')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(parent_folder, f'Kahler_form_loss_histogram.png'))
        plt.close()


        number_of_bins = 1000
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

        # Define the width of each bar
        width = 2 * np.pi / number_of_bins
    
        # --- Calculate counts first to determine the baseline ---
        counts_A, bin_edges_A = np.histogram(phases, bins=number_of_bins, range=(0, 2 * np.pi))
        angles_A = bin_edges_A[:-1]

        counts_B, bin_edges_B = np.histogram(phases_random, bins=number_of_bins, range=(0, 2 * np.pi))
        angles_B = bin_edges_B[:-1]

        # --- Set baseline dynamically to half the max peak height ---
        max_count = counts_A.max()
        baseline_radius = max_count / 2

        # Plot the bars with the new baseline
        ax.bar(angles_A, counts_A, width=width, alpha=0.7, color='skyblue', label='Potential sLag', bottom=baseline_radius)
        ax.bar(angles_B, counts_B, width=width, alpha=0.7, color='orange', label='Random intersection', bottom=baseline_radius)
        #ax.bar(angles_C, counts_C, width=width, alpha=0.7, color='#4CAF50', label='RP^3 with perturbation', bottom=baseline_radius)
        outer_limit = baseline_radius + max_count * 1.05


        # --- Format the plot ---
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)

        # --- Set angle labels to RADIANS ---
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2'], fontsize=12)

        # Set radial grid lines at 75% and 100% of the max height
        radial_grid_values = [baseline_radius + max_count * 0.25, baseline_radius + max_count * 0.5, baseline_radius + max_count*0.75]
        ax.set_rgrids(radial_grid_values, angle=22.5)
        ax.set_yticklabels([]) # Hide the number labels on the grid
        ax.grid(True, linestyle='--', alpha=0.6) # Keep grid but make it faint

        # Adjust the plot's outer limit to fit the data perfectly
        ax.set_rlim(0, baseline_radius + max_count * 1.05)


        ax.set_title('Distribution of the phases of the holomorphic 3-form', fontsize=16, pad=25)
        ax.legend(bbox_to_anchor=(1.1, 1.05))

        plt.savefig(os.path.join(parent_folder, f'circular_phase_histogram.png'), bbox_inches='tight')

    make_scatter_plots(min_set_real, parent_folder)
    save_min_set(min_set_real, parent_folder)

def make_scatter_plots(min_set_real: jnp.ndarray, parent_folder: str):

    min_set_x1 = min_set_real[:, 1]
    min_set_x2 = min_set_real[:, 2]

    plt.figure(figsize=(10, 6))
    plt.scatter(min_set_x2, min_set_x1, alpha=1.0, color='black', edgecolor='black',s=0.2)
    plt.title('Scatter Plot of z1 real vs z2 real')
    plt.xlabel('z2 real')
    plt.ylabel('z1 real')
    plt.grid(True, linestyle='--', alpha=0.6)
    output_filename = os.path.join(parent_folder, 'scatter_plot_z1rz2r.png')
    plt.savefig(output_filename, dpi=300)
    plt.close()

    min_set_x1 = min_set_real[:, 3]
    min_set_x2 = min_set_real[:, 4]

    plt.figure(figsize=(10, 6)) 
    plt.scatter(min_set_x2, min_set_x1, alpha=1.0, color='black', edgecolor='black',s=0.2)
    plt.title('Scatter Plot of z3 real vs z4 real')
    plt.xlabel('z4 real')
    plt.ylabel('z3 real')
    plt.grid(True, linestyle='--', alpha=0.6)
    output_filename = os.path.join(parent_folder, 'scatter_plot_z3rz4r.png')
    plt.savefig(output_filename, dpi=300)
    plt.close()


    min_set_x1 = min_set_real[:, 6]
    min_set_x2 = min_set_real[:, 7]

    plt.figure(figsize=(10, 6)) 
    plt.scatter(min_set_x2, min_set_x1, alpha=1.0, color='black', edgecolor='black',s=0.2)
    plt.title('Scatter Plot of z1 img vs z2 imag')
    plt.xlabel('z2 imag')
    plt.ylabel('z1 imag')
    plt.grid(True, linestyle='--', alpha=0.6)
    output_filename = os.path.join(parent_folder, 'scatter_plot_z1iz2i.png')
    plt.savefig(output_filename, dpi=300)
    plt.close()


    min_set_x1 = min_set_real[:, 8]
    min_set_x2 = min_set_real[:, 9]

    plt.figure(figsize=(10, 6)) 
    plt.scatter(min_set_x2, min_set_x1, alpha=1.0, color='black', edgecolor='black',s=0.2)
    plt.title('Scatter Plot of z3 img vs z4 imag')
    plt.xlabel('z4 imag')
    plt.ylabel('z3 imag')
    plt.grid(True, linestyle='--', alpha=0.6)
    output_filename = os.path.join(parent_folder, 'scatter_plot_z3iz4i.png')
    plt.savefig(output_filename, dpi=300)
    plt.close()


def save_min_set(min_set_real: jnp.ndarray, parent_folder: str) -> None:
    min_set = min_set_real[:,:5]+min_set_real[:,5:]*1j
    with open(os.path.join(parent_folder, "min_set.pkl"), "wb") as f:
        pickle.dump(min_set, f) 
