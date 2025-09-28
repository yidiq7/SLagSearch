import jax
import jax.numpy as jnp
import numpy as np
import os
import re
import pickle
import matplotlib.pyplot as plt
from find_smooth_submanifold import filter_and_refine, normalize_coeffs
from slag_condition import compute_combined_fitness
from helper import canonicalize_coeffs 
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D 

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
    print(f"min_set_distance: Min: {jnp.min(distances)}, Max: {jnp.max(distances)}, Mean: {jnp.mean(distances)}")
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
    plt.scatter(min_set_x1, min_set_x2, alpha=1.0, color='black', edgecolor='black',s=0.2)
    plt.title('Scatter Plot of z2 real vs z1 real')
    plt.xlabel('z1 real')
    plt.ylabel('z2 real')
    plt.grid(True, linestyle='--', alpha=0.6)
    output_filename = os.path.join(parent_folder, 'scatter_plot_z1rz2r.png')
    plt.savefig(output_filename, dpi=300)
    plt.close()

    min_set_x1 = min_set_real[:, 3]
    min_set_x2 = min_set_real[:, 4]

    plt.figure(figsize=(10, 6)) 
    plt.scatter(min_set_x1, min_set_x2, alpha=1.0, color='black', edgecolor='black',s=0.2)
    plt.title('Scatter Plot of z4 real vs z3 real')
    plt.xlabel('z3 real')
    plt.ylabel('z4 real')
    plt.grid(True, linestyle='--', alpha=0.6)
    output_filename = os.path.join(parent_folder, 'scatter_plot_z3rz4r.png')
    plt.savefig(output_filename, dpi=300)
    plt.close()


    min_set_x1 = min_set_real[:, 6]
    min_set_x2 = min_set_real[:, 7]

    plt.figure(figsize=(10, 6)) 
    plt.scatter(min_set_x1, min_set_x2, alpha=1.0, color='black', edgecolor='black',s=0.2)
    plt.title('Scatter Plot of z2 img vs z1 imag')
    plt.xlabel('z1 imag')
    plt.ylabel('z2 imag')
    plt.grid(True, linestyle='--', alpha=0.6)
    output_filename = os.path.join(parent_folder, 'scatter_plot_z1iz2i.png')
    plt.savefig(output_filename, dpi=300)
    plt.close()


    min_set_x1 = min_set_real[:, 8]
    min_set_x2 = min_set_real[:, 9]

    plt.figure(figsize=(10, 6)) 
    plt.scatter(min_set_x1, min_set_x2, alpha=1.0, color='black', edgecolor='black',s=0.2)
    plt.title('Scatter Plot of z4 img vs z3 imag')
    plt.xlabel('z3 imag')
    plt.ylabel('z4 imag')
    plt.grid(True, linestyle='--', alpha=0.6)
    output_filename = os.path.join(parent_folder, 'scatter_plot_z3iz4i.png')
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


'''
def plot_slag_data(jobid, max_rank, coordinates):
    """
    Finds, loads, and plots data from GA output folders.
    """

    coord_list = [
        'z0_real',
        'z1_real', 
        'z2_real',  
        'z3_real',  
        'z4_real',  
        'z0_img',   
        'z1_img',   
        'z2_img',   
        'z3_img',   
        'z4_img'    
    ]


    main_folder = f"plots_slag_{jobid}"

    if not os.path.isdir(main_folder):
        print(f"Error: Base directory '{main_folder}' not found.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    all_points_x = []
    all_points_y = []

    # Regex to parse folder names like 'plots_slag_12345_5_id678'
    pattern = re.compile(f"plots_slag_{jobid}_(\\d+)_id\\d+")

    print(f"Searching for subfolders in '{main_folder}' with rank < {max_rank}...")

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
                                # Load the pickled data (assuming it's a jnp or numpy array)
                                min_set_complex = pickle.load(f)

                            # Convert jax array to a standard numpy array for processing
                            min_set_complex_np = np.asarray(min_set_complex)

                            # Convert the N x 5 complex array to an N x 10 real array
                            # by stacking the real and imaginary parts
                            min_set_real = np.hstack([min_set_complex_np.real, min_set_complex_np.imag])

                            # Append the second (index 1) and third (index 2) coordinates
                            all_points_y.extend(min_set_real[:, coordinates[0]])
                            all_points_x.extend(min_set_real[:, coordinates[1]])

                        except Exception as e:
                            print(f"Warning: Could not process file {pkl_path}. Error: {e}")

    if all_points_x and all_points_y:
        print(f"Plotting {len(all_points_x)} total points.")
        ax.scatter(all_points_x, all_points_y, s=0.1, alpha=0.7, edgecolors='none')
        ax.set_title(f"{coord_list[coordinates[0]]} vs. {coord_list[coordinates[1]]} for Job ID: {jobid} (Ranks < {max_rank})", fontsize=16)
        ax.set_xlabel(f"{coord_list[coordinates[1]]}", fontsize=12)
        ax.set_ylabel(f"{coord_list[coordinates[1]]}", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()
        plt.savefig(os.path.join(main_folder, f'scatter_plot_{jobid}_{coordinates[0]}_{coordinates[1]}.png'))
        #plt.show()
    else:
        print("No data found to plot. Check folder names and paths.")

'''
