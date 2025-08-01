import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import zoom
from skimage import draw
from datetime import datetime
import imageio


def plot_point_correspondences(points1, points2, labels=None, colors=None, 
                              line_alpha=0.3, point_size=30, figsize=(12, 8),
                              title="Point Correspondences", show_indices=False):
    """
    Plot two sets of points with lines connecting corresponding points by index.
    
    Parameters:
    -----------
    points1 : numpy.ndarray
        First set of points, shape (N, 2) with (x, y) coordinates
    points2 : numpy.ndarray  
        Second set of points, shape (N, 2) with (x, y) coordinates
    labels : tuple, optional
        Labels for the two point sets, e.g., ('Set A', 'Set B')
    colors : tuple, optional
        Colors for the two point sets, e.g., ('red', 'blue')
    line_alpha : float, default=0.3
        Transparency of correspondence lines
    point_size : int, default=30
        Size of the plotted points
    figsize : tuple, default=(12, 8)
        Figure size
    title : str, default="Point Correspondences"
        Plot title
    show_indices : bool, default=False
        Whether to show point indices as text annotations
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Ensure points are numpy arrays
    points1 = np.array(points1)
    points2 = np.array(points2)
    
    # Check that both sets have the same number of points
    if len(points1) != len(points2):
        print(f"Warning: Point sets have different lengths ({len(points1)} vs {len(points2)})")
        min_len = min(len(points1), len(points2))
        points1 = points1[:min_len]
        points2 = points2[:min_len]
        print(f"Using first {min_len} points from each set")
    
    # Set default labels and colors
    if labels is None:
        labels = ('Points Set 1', 'Points Set 2')
    if colors is None:
        colors = ('red', 'blue')
    
    # Create the plot
    fig = plt.figure(figsize=figsize)
    
    # Draw correspondence lines first (so they appear behind points)
    for i in range(len(points1)):
        plt.plot([points1[i, 0], points2[i, 0]], 
                [points1[i, 1], points2[i, 1]], 
                'gray', alpha=line_alpha, linewidth=1, zorder=1)
    
    # Plot the points
    scatter1 = plt.scatter(points1[:, 0], points1[:, 1], 
                         c=colors[0], s=point_size, alpha=0.8, 
                         label=labels[0], zorder=3, edgecolors='black', linewidth=0.5)
    scatter2 = plt.scatter(points2[:, 0], points2[:, 1], 
                         c=colors[1], s=point_size, alpha=0.8, 
                         label=labels[1], zorder=3, edgecolors='black', linewidth=0.5)
    
    # Add point indices if requested
    if show_indices:
        for i in range(len(points1)):
            plt.annotate(f'{i}', (points1[i, 0], points1[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7, color=colors[0])
            plt.annotate(f'{i}', (points2[i, 0], points2[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7, color=colors[1])
    
    # Formatting
    #ax.set_xlabel('X coordinate')
    #ax.set_ylabel('Y coordinate')
    plt.title(title, fontsize=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    #plt.xlim(0, 256)
    #plt.ylim(0, 256)
    #plt.axis('off')
    #plt.gca().set_aspect('equal', adjustable='box')
    
    # Print statistics
    #print(f"Plotted {len(points1)} point correspondences")
    #print(f"Set 1 - X range: [{points1[:, 0].min():.1f}, {points1[:, 0].max():.1f}], "
    #      f"Y range: [{points1[:, 1].min():.1f}, {points1[:, 1].max():.1f}]")
    #print(f"Set 2 - X range: [{points2[:, 0].min():.1f}, {points2[:, 0].max():.1f}], "
    #      f"Y range: [{points2[:, 1].min():.1f}, {points2[:, 1].max():.1f}]")
    
    # Calculate and print correspondence statistics
    distances = np.sqrt(np.sum((points1 - points2)**2, axis=1))
    #print(f"Correspondence distances - Mean: {distances.mean():.2f}, "
    #      f"Std: {distances.std():.2f}, Max: {distances.max():.2f}")
    
    #plt.tight_layout()
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()
    
    return fig


def circle_to_ellipse_correspondences(
    n_points=30,
    circle_center=(100, 100),
    circle_radius=50,
    ellipse_offset=(150, 100),
    stretch_x=1.5,
    stretch_y=0.7,
    angle=np.pi/6,
    plot=True,
    point_size=40,
    show_indices=False,
    labels=('Circle', 'Rotated Ellipse'),
    colors=('purple', 'cyan'),
    title='Circle to Ellipse Transformation'
):
    """
    Generate corresponding points on a circle and a stretched/rotated ellipse.

    Returns:
        circle_points: (N, 2) array
        ellipse_points: (N, 2) array
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle_points = np.column_stack([
        circle_radius * np.cos(theta) + circle_center[0],
        circle_radius * np.sin(theta) + circle_center[1]
    ])

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ellipse_points = np.zeros_like(circle_points)
    for i, (x, y) in enumerate(circle_points - circle_center):  # Center at origin
        x_stretched = x * stretch_x
        y_stretched = y * stretch_y
        ellipse_points[i, 0] = x_stretched * cos_a - y_stretched * sin_a + ellipse_offset[0]
        ellipse_points[i, 1] = x_stretched * sin_a + y_stretched * cos_a + ellipse_offset[1]

    if plot:
        plot_point_correspondences(
            circle_points, ellipse_points,
            labels=labels,
            colors=colors,
            title=title,
            show_indices=show_indices,
            point_size=point_size
        )
    return circle_points, ellipse_points


def create_blobby_circular_shapes(n_points=30, noise_scale=4, seed=42, plot=True):
    """
    Create two blobby, mostly circular polygon-like shapes with corresponding points.
    Ensures the points do not intersect each other on the same shape.
    Both shapes are centered around the same region.
    Returns:
        shape1_points: (N, 2) array of integers
        shape2_points: (N, 2) array of integers
    """
    np.random.seed(seed)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Make radii mostly circular, with small smooth perturbations
    r1 = 50 + 4 * np.sin(2 * theta) + 2 * np.cos(3 * theta)
    r2 = 48 + 5 * np.sin(2 * theta + 0.5) + 3 * np.cos(4 * theta + 1.2)

    # Add small, smooth noise (not per-point, to avoid self-intersections)
    r1 += np.interp(theta, [0, 2*np.pi], np.random.normal(0, noise_scale, 2))
    r2 += np.interp(theta, [0, 2*np.pi], np.random.normal(0, noise_scale, 2))

    # Center both shapes at the same location
    center = np.array([128, 128])

    # Convert polar to cartesian and round to integers
    x1 = np.round(r1 * np.cos(theta) + center[0]).astype(int)
    y1 = np.round(r1 * np.sin(theta) + center[1]).astype(int)
    x2 = np.round(r2 * np.cos(theta + 0.18) + center[0]).astype(int)
    y2 = np.round(r2 * np.sin(theta + 0.18) + center[1]).astype(int)

    shape1_points = np.stack([x1, y1], axis=1)
    shape2_points = np.stack([x2, y2], axis=1)

    if plot:
        plt.figure(figsize=(8, 8))
        #plt.plot(*shape1_points.T, 'o-', color='purple', label='Shape 1')
        #plt.plot(*shape2_points.T, 'o-', color='orange', label='Shape 2')
        for i in range(n_points):
            plt.plot([shape1_points[i, 0], shape2_points[i, 0]],
                     [shape1_points[i, 1], shape2_points[i, 1]],
                     color='gray', alpha=0.4)
        plt.scatter(shape1_points[:, 0], shape1_points[:, 1], c='purple', s=60, edgecolors='k', zorder=3, label='Shape 1')
        plt.scatter(shape2_points[:, 0], shape2_points[:, 1], c='orange', s=60, edgecolors='k', zorder=3, label='Shape 2')
        plt.title("Blobby Circular Shapes with Corresponding Points (Centered)")
        plt.axis('equal')
        plt.legend()
        plt.show()

    return shape1_points, shape2_points


def create_sinusoidal_wave(grid_size=256, num_cycles=4, amplitude_ratio=0.25, 
                          vertical_offset_ratio=0.5, padding_ratio=0.1, 
                          phase_shift=0, plot=True):
    """
    Create a sinusoidal wave in a grid with customizable parameters.
    
    Parameters:
    -----------
    grid_size : int, default=256
        Size of the square grid (grid_size x grid_size)
    num_cycles : float, default=4
        Number of complete sine wave cycles across the grid
    amplitude_ratio : float, default=0.25
        Amplitude as a ratio of grid_size (0.25 = quarter of grid height)
    vertical_offset_ratio : float, default=0.5
        Vertical center position as ratio of grid_size (0.5 = center)
    padding_ratio : float, default=0.1
        Padding around the wave as ratio of grid_size (0.1 = 10% padding)
    phase_shift : float, default=0
        Phase shift in radians for the sine wave
    plot : bool, default=True
        Whether to create plots of the wave
        
    Returns:
    --------
    wave_points : numpy.ndarray
        Array of (x, y) coordinates representing points on the wave
    grid : numpy.ndarray
        2D grid with wave points marked as 1, empty space as 0
    """
    
    # Calculate actual wave parameters
    padding = int(grid_size * padding_ratio)
    effective_grid_size = grid_size - 2 * padding
    
    # Create x coordinates for the effective area (excluding padding)
    x_coords = np.arange(effective_grid_size) + padding
    
    # Calculate wave parameters
    frequency = 2 * np.pi * num_cycles / effective_grid_size
    amplitude = grid_size * amplitude_ratio
    vertical_offset = grid_size * vertical_offset_ratio
    
    # Calculate y coordinates for the sine wave
    x_wave = np.arange(effective_grid_size)  # Use 0-based for wave calculation
    y_coords = amplitude * np.sin(frequency * x_wave + phase_shift) + vertical_offset
    
    # Convert to integer coordinates and ensure they stay within bounds
    y_coords_int = np.round(y_coords).astype(int)
    y_coords_int = np.clip(y_coords_int, 0, grid_size - 1)
    
    # Create points that lie on the wave (x, y pairs)
    wave_points = np.column_stack((x_coords, y_coords_int))
    
    # Create a 2D grid visualization
    grid = np.zeros((grid_size, grid_size))
    for x, y in wave_points:
        grid[y, x] = 1
    
    # Print wave statistics
    print(f"Created {len(wave_points)} points on sinusoidal wave")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Padding: {padding} pixels on each side")
    print(f"Effective wave area: {effective_grid_size}x{grid_size}")
    print(f"Number of cycles: {num_cycles}")
    print(f"X range: {x_coords.min()} to {x_coords.max()}")
    print(f"Y range: {y_coords_int.min()} to {y_coords_int.max()}")
    print(f"Amplitude: {amplitude:.1f} pixels")
    print(f"First 5 points: {wave_points[:5]}")
    
    if plot:
        # Plot the sinusoidal wave
        plt.figure(figsize=(15, 10))
        
        # Plot the grid with the wave and padding visualization
        plt.subplot(2, 2, 1)
        # Create a colored grid to show padding
        colored_grid = np.ones((grid_size, grid_size)) * 0.3  # Gray for padding
        colored_grid[padding:grid_size-padding, padding:grid_size-padding] = 0  # White for effective area
        colored_grid += grid  # Add wave (blue)
        
        plt.imshow(colored_grid, cmap='Blues', origin='lower', extent=[0, grid_size, 0, grid_size])
        plt.title(f'Sinusoidal Wave on {grid_size}x{grid_size} Grid\n(Gray areas show padding)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.colorbar(label='Grid zones')
        
        # Plot just the wave points with grid
        plt.subplot(2, 2, 2)
        plt.plot(wave_points[:, 0], wave_points[:, 1], 'b-', linewidth=2, label='Sinusoidal wave')
        plt.scatter(wave_points[::max(1, len(wave_points)//20), 0], 
                   wave_points[::max(1, len(wave_points)//20), 1], 
                   c='red', s=30, zorder=5, label='Sample points')
        
        # Show padding boundaries
        plt.axvline(x=padding, color='gray', linestyle='--', alpha=0.7, label='Padding boundary')
        plt.axvline(x=grid_size-padding, color='gray', linestyle='--', alpha=0.7)
        
        plt.title(f'Wave Points (Padding: {padding} pixels)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, grid_size)
        plt.ylim(0, grid_size)
        
        # Close-up view of the wave
        plt.subplot(2, 2, 3)
        plt.plot(wave_points[:, 0], wave_points[:, 1], 'b-', linewidth=2, alpha=0.8)
        plt.scatter(wave_points[::max(1, len(wave_points)//15), 0], 
                   wave_points[::max(1, len(wave_points)//15), 1], 
                   c='red', s=20, alpha=0.8)
        plt.title('Close-up View of Wave')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, alpha=0.3)
        
        # Wave parameters visualization
        plt.subplot(2, 2, 4)
        x_smooth = np.linspace(padding, grid_size-padding, 1000)
        x_wave_smooth = np.linspace(0, effective_grid_size, 1000)
        y_smooth = amplitude * np.sin(frequency * x_wave_smooth + phase_shift) + vertical_offset
        
        plt.plot(x_smooth, y_smooth, 'g--', linewidth=2, alpha=0.7, label='Continuous wave')
        plt.plot(wave_points[:, 0], wave_points[:, 1], 'bo-', markersize=3, linewidth=1, label='Discrete points')
        plt.axhline(y=vertical_offset, color='red', linestyle=':', alpha=0.5, label='Center line')
        plt.axhline(y=vertical_offset + amplitude, color='orange', linestyle=':', alpha=0.5, label='Max amplitude')
        plt.axhline(y=vertical_offset - amplitude, color='orange', linestyle=':', alpha=0.5, label='Min amplitude')
        
        plt.title('Wave Parameters Visualization')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, grid_size)
        
        plt.tight_layout()
        plt.show()
    
    return wave_points, grid


def create_double_helix_waves(grid_size=256, num_cycles=4, amplitude_ratio=0.2, 
                             vertical_center_ratio=0.5, padding_ratio=0.1, 
                             phase_offset=np.pi, plot=True, horizontal=False):
    """
    Create two intersecting sinusoidal waves that form a double helix pattern.
    If horizontal=True, the helix is oriented horizontally (waves run vertically).
    If horizontal=False, the helix is oriented vertically (waves run horizontally, original behavior).

    Parameters:
    -----------
    grid_size : int, default=256
        Size of the square grid (grid_size x grid_size)
    num_cycles : float, default=4
        Number of complete sine wave cycles across the grid
    amplitude_ratio : float, default=0.2
        Amplitude as a ratio of grid_size 
    vertical_center_ratio : float, default=0.5
        Vertical center position as ratio of grid_size (0.5 = center)
    padding_ratio : float, default=0.1
        Padding around the waves as ratio of grid_size
    phase_offset : float, default=np.pi
        Phase difference between the two waves (π creates perfect interleaving)
    plot : bool, default=True
        Whether to create plots of the waves
    horizontal : bool, default=True
        If True, helix is horizontal (waves run vertically). If False, original vertical helix.

    Returns:
    --------
    wave1_points : numpy.ndarray
        Array of (x, y) coordinates for the first wave
    wave2_points : numpy.ndarray
        Array of (x, y) coordinates for the second wave
    grid : numpy.ndarray
        2D grid with both waves marked (wave1=1, wave2=2, intersections=3)
    """

    padding = int(grid_size * padding_ratio)
    effective_grid_size = grid_size - 2 * padding

    if horizontal:
        # Swap axes: now y is the main axis, x is the wave
        y_coords = np.arange(effective_grid_size) + padding
        frequency = 2 * np.pi * num_cycles / effective_grid_size
        amplitude = grid_size * amplitude_ratio
        horizontal_center = grid_size * vertical_center_ratio

        y_wave = np.arange(effective_grid_size)
        # First wave (standard sine, runs horizontally)
        x1_coords = amplitude * np.sin(frequency * y_wave) + horizontal_center
        # Second wave (phase shifted sine)
        x2_coords = amplitude * np.sin(frequency * y_wave + phase_offset) + horizontal_center

        # Convert to integer coordinates and ensure they stay within bounds
        x1_coords_int = np.round(x1_coords).astype(int)
        x2_coords_int = np.round(x2_coords).astype(int)
        x1_coords_int = np.clip(x1_coords_int, 0, grid_size - 1)
        x2_coords_int = np.clip(x2_coords_int, 0, grid_size - 1)

        # Create points that lie on the waves (x, y)
        wave1_points = np.column_stack((x1_coords_int, y_coords))
        wave2_points = np.column_stack((x2_coords_int, y_coords))
    else:
        # Original vertical helix
        x_coords = np.arange(effective_grid_size) + padding
        frequency = 2 * np.pi * num_cycles / effective_grid_size
        amplitude = grid_size * amplitude_ratio
        vertical_center = grid_size * vertical_center_ratio

        x_wave = np.arange(effective_grid_size)
        y1_coords = amplitude * np.sin(frequency * x_wave) + vertical_center
        y2_coords = amplitude * np.sin(frequency * x_wave + phase_offset) + vertical_center

        y1_coords_int = np.round(y1_coords).astype(int)
        y2_coords_int = np.round(y2_coords).astype(int)
        y1_coords_int = np.clip(y1_coords_int, 0, grid_size - 1)
        y2_coords_int = np.clip(y2_coords_int, 0, grid_size - 1)

        wave1_points = np.column_stack((x_coords, y1_coords_int))
        wave2_points = np.column_stack((x_coords, y2_coords_int))

    # Create a 2D grid visualization
    grid = np.zeros((grid_size, grid_size))

    # Mark wave 1 points
    for x, y in wave1_points:
        grid[y, x] += 1

    # Mark wave 2 points 
    for x, y in wave2_points:
        grid[y, x] += 2

    # Find intersection points (where both waves occupy the same pixel)
    intersections = []
    for i, (x1, y1) in enumerate(wave1_points):
        for j, (x2, y2) in enumerate(wave2_points):
            if x1 == x2 and y1 == y2:
                intersections.append((i, j, x1, y1))

    #print(f"Created double helix with {len(wave1_points)} points per wave")
    #print(f"Grid size: {grid_size}x{grid_size}")
    #print(f"Padding: {padding} pixels on each side")
    #print(f"Number of cycles: {num_cycles}")
    #print(f"Phase offset: {phase_offset:.3f} radians ({phase_offset*180/np.pi:.1f} degrees)")
    #if horizontal:
        #print(f"Wave 1 - Y range: [{wave1_points[:,1].min()}, {wave1_points[:,1].max()}], X range: [{wave1_points[:,0].min()}, {wave1_points[:,0].max()}]")
        #print(f"Wave 2 - Y range: [{wave2_points[:,1].min()}, {wave2_points[:,1].max()}], X range: [{wave2_points[:,0].min()}, {wave2_points[:,0].max()}]")
    #else:
        #print(f"Wave 1 - X range: [{wave1_points[:,0].min()}, {wave1_points[:,0].max()}], Y range: [{wave1_points[:,1].min()}, {wave1_points[:,1].max()}]")
        #print(f"Wave 2 - X range: [{wave2_points[:,0].min()}, {wave2_points[:,0].max()}], Y range: [{wave2_points[:,1].min()}, {wave2_points[:,1].max()}]")
    #print(f"Number of intersection points: {len(intersections)}")

    if plot:
        fig = plt.figure(figsize=(18, 12))

        # Plot 1: Grid view showing both waves
        plt.subplot(2, 3, 1)
        display_grid = np.zeros((grid_size, grid_size, 3))  # RGB

        for x, y in wave1_points:
            display_grid[y, x, 0] = 1.0  # Red channel
        for x, y in wave2_points:
            display_grid[y, x, 2] = 1.0  # Blue channel

        plt.imshow(display_grid, origin='lower', extent=[0, grid_size, 0, grid_size])
        plt.title(f'Double Helix Grid View\n(Red: Wave 1, Blue: Wave 2, Purple: Intersections)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')

        # Plot 2: Both waves as line plots
        plt.subplot(2, 3, 2)
        plt.plot(wave1_points[:, 0], wave1_points[:, 1], 'r-', linewidth=2, label='Wave 1', alpha=0.8)
        plt.plot(wave2_points[:, 0], wave2_points[:, 1], 'b-', linewidth=2, label='Wave 2', alpha=0.8)
        if intersections:
            intersection_x = [pt[2] for pt in intersections]
            intersection_y = [pt[3] for pt in intersections]
            plt.scatter(intersection_x, intersection_y, c='purple', s=80, 
                       marker='o', label=f'Intersections ({len(intersections)})', 
                       zorder=5, edgecolors='black')
        plt.title('Double Helix Line View')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: 3D-like perspective (using offset)
        plt.subplot(2, 3, 3)
        offset = 10
        if horizontal:
            plt.plot(wave1_points[:, 0], wave1_points[:, 1], 'r-', linewidth=3, label='Wave 1 (front)', alpha=0.9)
            plt.plot(wave2_points[:, 0] + offset, wave2_points[:, 1], 'b-', linewidth=3, label='Wave 2 (back)', alpha=0.7)
        else:
            plt.plot(wave1_points[:, 0], wave1_points[:, 1], 'r-', linewidth=3, label='Wave 1 (front)', alpha=0.9)
            plt.plot(wave2_points[:, 0] + offset, wave2_points[:, 1], 'b-', linewidth=3, label='Wave 2 (back)', alpha=0.7)
        plt.title('Pseudo-3D View\n(Offset for depth perception)')
        plt.xlabel('X coordinate (+ offset)')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: Continuous waves for comparison
        plt.subplot(2, 3, 4)
        smooth = np.linspace(padding, grid_size-padding, 1000)
        smooth_wave = np.linspace(0, effective_grid_size, 1000)
        if horizontal:
            x1_smooth = amplitude * np.sin(frequency * smooth_wave) + horizontal_center
            x2_smooth = amplitude * np.sin(frequency * smooth_wave + phase_offset) + horizontal_center
            plt.plot(x1_smooth, smooth, 'r--', linewidth=2, alpha=0.7, label='Wave 1 (continuous)')
            plt.plot(x2_smooth, smooth, 'b--', linewidth=2, alpha=0.7, label='Wave 2 (continuous)')
            plt.plot(wave1_points[:, 0], wave1_points[:, 1], 'ro', markersize=3, alpha=0.6, label='Wave 1 (discrete)')
            plt.plot(wave2_points[:, 0], wave2_points[:, 1], 'bo', markersize=3, alpha=0.6, label='Wave 2 (discrete)')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
        else:
            y1_smooth = amplitude * np.sin(frequency * smooth_wave) + vertical_center
            y2_smooth = amplitude * np.sin(frequency * smooth_wave + phase_offset) + vertical_center
            plt.plot(smooth, y1_smooth, 'r--', linewidth=2, alpha=0.7, label='Wave 1 (continuous)')
            plt.plot(smooth, y2_smooth, 'b--', linewidth=2, alpha=0.7, label='Wave 2 (continuous)')
            plt.plot(wave1_points[:, 0], wave1_points[:, 1], 'ro', markersize=3, alpha=0.6, label='Wave 1 (discrete)')
            plt.plot(wave2_points[:, 0], wave2_points[:, 1], 'bo', markersize=3, alpha=0.6, label='Wave 2 (discrete)')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
        plt.title('Continuous vs Discrete Waves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 5: Phase relationship
        plt.subplot(2, 3, 5)
        sample = smooth_wave[:500]
        if horizontal:
            x1_sample = amplitude * np.sin(frequency * sample) + horizontal_center
            x2_sample = amplitude * np.sin(frequency * sample + phase_offset) + horizontal_center
            plt.plot(x1_sample, sample, 'r-', linewidth=3, label='Wave 1')
            plt.plot(x2_sample, sample, 'b-', linewidth=3, label='Wave 2')
            plt.axvline(x=horizontal_center, color='gray', linestyle=':', alpha=0.5, label='Center line')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
        else:
            y1_sample = amplitude * np.sin(frequency * sample) + vertical_center
            y2_sample = amplitude * np.sin(frequency * sample + phase_offset) + vertical_center
            plt.plot(sample, y1_sample, 'r-', linewidth=3, label='Wave 1')
            plt.plot(sample, y2_sample, 'b-', linewidth=3, label='Wave 2')
            plt.axhline(y=vertical_center, color='gray', linestyle=':', alpha=0.5, label='Center line')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
        plt.title(f'Phase Relationship\n(Phase offset: {phase_offset*180/np.pi:.1f}°)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 6: Intersection analysis
        plt.subplot(2, 3, 6)
        if intersections:
            intersection_indices = [pt[0] for pt in intersections]
            if horizontal:
                plt.scatter(intersection_indices, [wave1_points[i, 1] for i in intersection_indices], 
                            c='purple', s=60, alpha=0.8, label='Intersection Y positions')
                plt.xlabel('Wave 1 point index')
                plt.ylabel('Y coordinate of intersection')
            else:
                plt.scatter(intersection_indices, [wave1_points[i, 0] for i in intersection_indices], 
                            c='purple', s=60, alpha=0.8, label='Intersection X positions')
                plt.xlabel('Wave 1 point index')
                plt.ylabel('X coordinate of intersection')
            plt.title('Intersection Distribution')
            plt.grid(True, alpha=0.3)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No intersections found\nat current resolution', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Intersection Analysis')

        plt.tight_layout()
        plt.show()

    return wave1_points, wave2_points, grid


def create_ellipse_correspondences(n_points=30, center=(128, 128), 
                                   axes1=(40, 80), axes2=(40, 120), 
                                   angle1=0, angle2=0, seed=42, plot=True):
    """
    Create two ellipses with corresponding points, both centered at the same location.
    The second ellipse is longer along the y-axis.
    Returns:
        ellipse1_points: (N, 2) array of integers
        ellipse2_points: (N, 2) array of integers
    """
    np.random.seed(seed)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # First ellipse (shorter y-axis)
    x1 = axes1[0] * np.cos(theta)
    y1 = axes1[1] * np.sin(theta)
    # Second ellipse (longer y-axis)
    x2 = axes2[0] * np.cos(theta)
    y2 = axes2[1] * np.sin(theta)

    # Optionally rotate ellipses (angle in radians)
    def rotate(x, y, angle):
        xr = x * np.cos(angle) - y * np.sin(angle)
        yr = x * np.sin(angle) + y * np.cos(angle)
        return xr, yr

    x1, y1 = rotate(x1, y1, angle1)
    x2, y2 = rotate(x2, y2, angle2)

    # Center both ellipses
    x1 = np.round(x1 + center[0]).astype(int)
    y1 = np.round(y1 + center[1]).astype(int)
    x2 = np.round(x2 + center[0]).astype(int)
    y2 = np.round(y2 + center[1]).astype(int)

    ellipse1_points = np.stack([x1, y1], axis=1)
    ellipse2_points = np.stack([x2, y2], axis=1)

    if plot:
        plt.figure(figsize=(8, 8))
        plt.plot(*ellipse1_points.T, 'o-', color='blue', label='Ellipse 1 (short y)')
        plt.plot(*ellipse2_points.T, 'o-', color='red', label='Ellipse 2 (long y)')
        for i in range(n_points):
            plt.plot([ellipse1_points[i, 0], ellipse2_points[i, 0]],
                     [ellipse1_points[i, 1], ellipse2_points[i, 1]],
                     color='gray', alpha=0.4)
        plt.scatter(ellipse1_points[:, 0], ellipse1_points[:, 1], c='blue', s=60, edgecolors='k', zorder=3)
        plt.scatter(ellipse2_points[:, 0], ellipse2_points[:, 1], c='red', s=60, edgecolors='k', zorder=3)
        plt.title("Ellipse Correspondences (Same Center, Different Y-Axis)")
        plt.axis('equal')
        plt.legend()
        plt.show()

    return ellipse1_points, ellipse2_points


def create_crossing_lines_correspondences(n_points=80, center=(128, 128), length=180, plot=True):
    """
    Create two long crossing lines forming an X, with corresponding points.
    Both lines are centered at the same location.
    Returns:
        line1_points: (N, 2) array of integers
        line2_points: (N, 2) array of integers
    """
    # Line 1: from top-left to bottom-right
    x1 = np.linspace(center[0] - length // 2, center[0] + length // 2, n_points)
    y1 = np.linspace(center[1] - length // 2, center[1] + length // 2, n_points)
    # Line 2: from bottom-left to top-right
    x2 = np.linspace(center[0] - length // 2, center[0] + length // 2, n_points)
    y2 = np.linspace(center[1] + length // 2, center[1] - length // 2, n_points)

    # Round to integer pixel coordinates
    x1 = np.round(x1).astype(int)
    y1 = np.round(y1).astype(int)
    x2 = np.round(x2).astype(int)
    y2 = np.round(y2).astype(int)

    line1_points = np.stack([x1, y1], axis=1)
    line2_points = np.stack([x2, y2], axis=1)

    if plot:
        plt.figure(figsize=(8, 8))
        plt.plot(line1_points[:, 0], line1_points[:, 1], 'o-', color='blue', label='Line 1')
        plt.plot(line2_points[:, 0], line2_points[:, 1], 'o-', color='red', label='Line 2')
        for i in range(n_points):
            plt.plot([line1_points[i, 0], line2_points[i, 0]],
                     [line1_points[i, 1], line2_points[i, 1]],
                     color='gray', alpha=0.4)
        plt.scatter(line1_points[:, 0], line1_points[:, 1], c='blue', s=60, edgecolors='k', zorder=3)
        plt.scatter(line2_points[:, 0], line2_points[:, 1], c='red', s=60, edgecolors='k', zorder=3)
        plt.title("Crossing Lines (X) Correspondences")
        plt.axis('equal')
        plt.legend()
        plt.show()

    return line1_points, line2_points


# Load png function
def load_png(file_path: str, scale: float = 1.0, alpha: float = 1.0):
    """Load a color PNG image and return it as a numpy array, with scaling and alpha."""
    img_array = mpimg.imread(file_path)  # shape: (H, W, 3) or (H, W, 4)
    if img_array.ndim == 2:  # Grayscale image
        img_array = np.stack((img_array,) * 3, axis=-1)  # Convert to RGB
    # Only scale height and width, not channels
    if scale != 1.0:
        img_array = zoom(img_array, (scale, scale, 1), order=1)
    # If overlay is not RGBA, convert to RGBA
    if img_array.shape[-1] == 3:
        alpha_channel = np.ones(img_array.shape[:2], dtype=img_array.dtype) * alpha
        img_array = np.dstack((img_array, alpha_channel))
    else:
        img_array[..., 3] = alpha  # Set alpha for all pixels
    return img_array

    
def show_2_pngs(image1, image2, title1=None, title2=None):
    """Display two color images side by side using matplotlib."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    if image1.shape[-1] == 4:  # RGBA
        rgb1 = image1[..., :3]
        alpha1 = image1[..., 3:4]
        white1 = np.ones_like(rgb1)
        img1 = alpha1 * rgb1 + (1 - alpha1) * white1
    else:
        img1 = image1
    
    if image2.shape[-1] == 4:  # RGBA
        rgb2 = image2[..., :3]
        alpha2 = image2[..., 3:4]
        white2 = np.ones_like(rgb2)
        img2 = alpha2 * rgb2 + (1 - alpha2) * white2
    else:
        img2 = image2
    
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_raw_image(points, shape=(256, 256), save_path=None):
    """
    Create a binary image (white background, black points) from a list of (z, y, x) or (y, x) points.
    Optionally save the image to disk.

    Args:
        points: np.ndarray of shape (N, 2) or (N, 3). If (N, 3), uses points[:, 1], points[:, 2] as (y, x).
        shape: tuple, image shape (height, width).
        save_path: str or None. If provided, saves the image as a PNG.

    Returns:
        img: np.ndarray, binary image.
    """
    img = np.ones(shape, dtype=np.float32)
    if points.shape[1] == 3:
        ys, xs = points[:, 1], points[:, 2]
    else:
        ys, xs = points[:, 0], points[:, 1]
    ys = np.round(ys).astype(int)
    xs = np.round(xs).astype(int)
    valid = (ys >= 0) & (ys < shape[0]) & (xs >= 0) & (xs < shape[1])
    img[ys[valid], xs[valid]] = 0

    if save_path is not None:
        imageio.imwrite(save_path, (img * 255).astype(np.uint8))
    return img