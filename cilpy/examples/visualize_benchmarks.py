import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List

from cilpy.problem.mpb import MovingPeaksBenchmark, generate_mpb_configs
from cilpy.problem.cmpb import ConstrainedMovingPeaksBenchmark

# --- Visualization Functions ---

def get_current_landscape(problem: MovingPeaksBenchmark, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """Calculates a snapshot of the MPB fitness landscape."""
    if problem.dimension != 2:
        raise ValueError("Visualization is only supported for 2D problems.")
    fitness_grid = np.zeros_like(x_coords)
    for i in range(x_coords.shape[0]):
        for j in range(x_coords.shape[1]):
            solution = np.array([x_coords[i, j], y_coords[i, j]])
            peak_values = [p.evaluate(solution) for p in problem.peaks]
            max_value = float(max([0.0] + peak_values))
            fitness_grid[i, j] = -max_value
    return fitness_grid

def get_current_cmpb_landscape(problem: ConstrainedMovingPeaksBenchmark, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """Calculates a snapshot of the CMPB constraint violation landscape."""
    if problem.dimension != 2:
        raise ValueError("Visualization is only supported for 2D problems.")
    violation_grid = np.zeros_like(x_coords)
    f_landscape = problem.f_landscape
    g_landscape = problem.g_landscape
    for i in range(x_coords.shape[0]):
        for j in range(x_coords.shape[1]):
            solution = np.array([x_coords[i, j], y_coords[i, j]])
            f_val = float(max([0.0] + [p.evaluate(solution) for p in f_landscape.peaks]))
            g_val = float(max([0.0] + [p.evaluate(solution) for p in g_landscape.peaks]))
            violation_grid[i, j] = g_val - f_val
    return violation_grid

def main():
    """
    Generates and saves a single image showing three consecutive
    environment changes for an MPB or CMPB problem.
    """
    # ----------------------------------------------------------------------
    # 1. --- CONFIGURATION ---
    # ----------------------------------------------------------------------
    # Set to True for CMPB, False for standard MPB
    VISUALIZE_CONSTRAINED = True
    
    # Number of consecutive environments to display
    NUM_ENVIRONMENTS = 3
    
    # Use a chaotic configuration for interesting, unpredictable changes
    all_configs = generate_mpb_configs(dimension=2)
    config_name = 'C3R' # Chaotic, Type III, Random
    params = all_configs[config_name]

    # Set a change frequency. This is how many evaluations trigger a change.
    CHANGE_FREQUENCY = 100
    params['change_frequency'] = CHANGE_FREQUENCY

    # ----------------------------------------------------------------------
    # 2. --- PROBLEM INITIALIZATION ---
    # ----------------------------------------------------------------------
    if VISUALIZE_CONSTRAINED:
        print("Mode: Constrained Moving Peaks Benchmark (CMPB)")
        # For CMPB, we need two sets of parameters
        f_params = all_configs['A1L'] # Objective: Abrupt, Type I, Linear
        g_params = all_configs['C3R'] # Constraint: Chaotic, Type III, Random
        f_params['change_frequency'] = CHANGE_FREQUENCY
        g_params['change_frequency'] = CHANGE_FREQUENCY
        
        problem = ConstrainedMovingPeaksBenchmark(f_params, g_params)
        output_filename = "cmpb_consecutive_changes.jpeg"
    else:
        print("Mode: Standard Moving Peaks Benchmark (MPB)")
        problem = MovingPeaksBenchmark(**params)
        output_filename = "mpb_consecutive_changes.jpeg"

    # ----------------------------------------------------------------------
    # 3. --- DATA GENERATION ---
    # ----------------------------------------------------------------------
    # Create the coordinate grid
    grid_points = 150
    domain_min, domain_max = problem.bounds[0][0], problem.bounds[1][0]
    x = np.linspace(domain_min, domain_max, grid_points)
    y = np.linspace(domain_min, domain_max, grid_points)
    X, Y = np.meshgrid(x, y)
    
    landscapes: List[np.ndarray] = []
    print(f"Generating {NUM_ENVIRONMENTS} landscape snapshots...")

    for i in range(NUM_ENVIRONMENTS):
        print(f" - Capturing state for Environment #{i+1}...")
        # Get a snapshot of the current landscape
        if VISUALIZE_CONSTRAINED:
            snapshot = get_current_cmpb_landscape(problem, X, Y) # type: ignore
        else:
            snapshot = get_current_landscape(problem, X, Y) # type: ignore
        landscapes.append(snapshot)
        
        # Advance the problem to the next environment state
        # (unless it's the last one)
        if i < NUM_ENVIRONMENTS - 1:
            dummy_solution = problem.bounds[0]
            for _ in range(CHANGE_FREQUENCY):
                problem.evaluate(dummy_solution)

    # ----------------------------------------------------------------------
    # 4. --- PLOTTING ---
    # ----------------------------------------------------------------------
    print("Creating plot...")
    fig, axes = plt.subplots(1, NUM_ENVIRONMENTS, figsize=(18, 6), sharey=True)
    
    # Determine global color limits for consistency
    global_min = min(z.min() for z in landscapes)
    global_max = max(z.max() for z in landscapes)
    
    im = None
    for i, Z in enumerate(landscapes):
        ax = axes[i]
        
        if VISUALIZE_CONSTRAINED:
            # Use custom blue-black-red colormap for CMPB
            custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_bkr", [(0, 0, 1), (0, 0, 0), (1, 0, 0)])
            norm = mcolors.TwoSlopeNorm(vmin=global_min, vcenter=0, vmax=global_max)
            im = ax.imshow(Z, cmap=custom_cmap, norm=norm, origin='lower', extent=[domain_min, domain_max, domain_min, domain_max])
            # im = ax.imshow(Z, cmap='magma', norm=norm, origin='lower', extent=[domain_min, domain_max, domain_min, domain_max])
        else:
            # Use custom blue-black colormap for MPB
            custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_bkr", [(0, 0, 1), (0, 0, 0)])
            im = ax.imshow(Z, cmap=custom_cmap, vmin=global_min, vmax=global_max, origin='lower', extent=[domain_min, domain_max, domain_min, domain_max])
            # im = ax.imshow(Z, cmap='magma', vmin=global_min, vmax=global_max, origin='lower', extent=[domain_min, domain_max, domain_min, domain_max])
            
        ax.set_title(f"Environment #{i + 1}", fontsize=14)
        ax.set_xlabel("Dimension 1")
        if i == 0:
            ax.set_ylabel("Dimension 2")

    # Add a single, shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    cbar = fig.colorbar(im, cax=cbar_ax) # type: ignore
    
    if VISUALIZE_CONSTRAINED:
        cbar.set_label("Constraint Violation (g(x) - f(x))", fontsize=12)
    else:
        cbar.set_label("Fitness Value", fontsize=12)

    # ----------------------------------------------------------------------
    # 5. --- SAVING AND DISPLAYING ---
    # ----------------------------------------------------------------------
    plt.savefig(output_filename, format="jpeg" ,dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as '{output_filename}'")
    plt.show()


if __name__ == '__main__':
    main()
