import jax
import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt
from find_smooth_submanifold import filter_and_refine, normalize_coeffs
from slag_condition import compute_combined_fitness
from helper import canonicalize_coeffs
from typing import Optional

def make_fitness_plots(
    points_real: jnp.ndarray,
    coeffs: jnp.ndarray,
    psi: jnp.ndarray,
    k: int = 100000,
    n_refine_steps: int = 100,
    constant_coord: int = 0,
    compare_with_random: bool = False,
    suffix: Optional[str] = None
    ):

    os.makedirs('plots_slag', exist_ok=True)
    kahler_form_path = 'plots_slag/Kahler_form_loss_histogram.png'
    phase_path = 'plots_slag/circular_phase_histogram.png'
    if suffix is not None:
        kahler_form_path = f"{kahler_form_path}{suffix}"
        phase_path = f"{phase_path}{suffix}"

    min_set_real, distances = filter_and_refine(points_real, coeffs, psi, k, n_refine_steps, constant_coord, debug_mode=True)
    total_fitness, lagrangian_fitness, special_fitness, kahler_form_restricted, restriction, phases = compute_combined_fitness(min_set_real, coeffs, psi, debug_mode=True)

    frobenius_norms = jnp.linalg.norm(kahler_form_restricted, axis=(1, 2))


    if not compare_with_random:
        # Plot the Kahler form loss
        plt.figure(figsize=(10, 6))
        plt.hist(frobenius_norms, bins=200, alpha=0.7, label='Potential sLag', color='skyblue', density=True)
        plt.xlabel('Frobenius norm')
        plt.ylabel('Counts')
        plt.title('Distribution of the norm of the Kahler form')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('plots_slag/Kahler_form_loss_histogram.png')
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

        plt.savefig('plots_slag/circular_phase_histogram.png', bbox_inches='tight')


     # The option to plot both slag and random manifold in one plot for comparison
    elif compare_with_random:
        seed = 1230
        key = jax.random.PRNGKey(seed)
        coeffs_random = jax.random.uniform(key, (3, 25), minval=-1, maxval=1)
        coeffs_random =  canonicalize_coeffs(coeffs_random)
        coeffs_random =  normalize_coeffs(coeffs_random)

        min_set_real_random, distances_random = filter_and_refine(points_real, coeffs_random, psi, k, n_refine_steps, constant_coord, debug_mode=True)
        total_fitness_random, lagrangian_fitness_random, special_fitness_random, kahler_form_restricted_random, restriction_random, phases_random = compute_combined_fitness(min_set_real_random, coeffs_random, psi, debug_mode=True)
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
        plt.savefig(kahler_form_path)
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

        plt.savefig(phase_path, bbox_inches='tight')


