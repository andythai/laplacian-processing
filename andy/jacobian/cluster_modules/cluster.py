import numpy as np
import matplotlib.pyplot as plt
from modules.data import Data


def compute_corresponding_pixels_with_inverse(data_obj, plot=True, title="Point Correspondences via Forward and Inverse Deformation"):
    """
    Compute corresponding pixels using both forward and inverse deformation fields.
    Creates an inverse Data object by swapping moving and fixed points, then compares results.
    
    Parameters:
    -----------
    data_obj : Data object
        The original Data object containing the forward deformation field
    plot : bool, default=True
        Whether to create visualization plots
    title : str
        Title for the plots
        
    Returns:
    --------
    forward_coords : numpy.ndarray
        Forward mapping: fixed coordinates mapped to moving space
    inverse_coords : numpy.ndarray  
        Inverse mapping: moving coordinates mapped to fixed space
    inverse_data_obj : Data object
        The inverse Data object (with swapped points)
    """
    
    # Get original data
    resolution = data_obj.resolution
    original_mpoints = data_obj.mpoints.copy()
    original_fpoints = data_obj.fpoints.copy()
    forward_deformation = data_obj.deformation.copy()
    
    print(f"Original Data object:")
    print(f"  Moving points shape: {original_mpoints.shape}")
    print(f"  Fixed points shape: {original_fpoints.shape}")
    print(f"  Resolution: {resolution}")
    print(f"  Forward deformation shape: {forward_deformation.shape}")
    
    # Create inverse Data object by swapping moving and fixed points
    print(f"\nCreating inverse Data object by swapping moving and fixed points...")
    inverse_data_obj = Data(original_fpoints, original_mpoints, resolution)
    inverse_deformation = inverse_data_obj.deformation.copy()
    
    print(f"Inverse Data object:")
    print(f"  Moving points (was fixed): {inverse_data_obj.mpoints.shape}")
    print(f"  Fixed points (was moving): {inverse_data_obj.fpoints.shape}")
    print(f"  Inverse deformation shape: {inverse_deformation.shape}")
    
    # Create coordinate grids
    height, width = resolution
    z_grid = np.zeros((height, width))
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    coord_grid = np.stack([z_grid, y_grid, x_grid], axis=0)
    
    # Forward mapping: Fixed -> Moving
    forward_coords = coord_grid + forward_deformation[:, 0, :, :]
    forward_coords_int = np.round(forward_coords).astype(int)
    
    # Inverse mapping: Moving -> Fixed  
    inverse_coords = coord_grid + inverse_deformation[:, 0, :, :]
    inverse_coords_int = np.round(inverse_coords).astype(int)
    
    print(f"\nDeformation field statistics:")
    print(f"Forward deformation - Y: [{forward_deformation[1, 0].min():.2f}, {forward_deformation[1, 0].max():.2f}]")
    print(f"Forward deformation - X: [{forward_deformation[2, 0].min():.2f}, {forward_deformation[2, 0].max():.2f}]")
    print(f"Inverse deformation - Y: [{inverse_deformation[1, 0].min():.2f}, {inverse_deformation[1, 0].max():.2f}]")
    print(f"Inverse deformation - X: [{inverse_deformation[2, 0].min():.2f}, {inverse_deformation[2, 0].max():.2f}]")
    
    # Apply inverse deformation to original moving points to get their fixed coordinates
    transformed_moving_points = []
    for i, mp in enumerate(original_mpoints):
        z_mp, y_mp, x_mp = mp.astype(int)
        
        # Make sure coordinates are within bounds
        if 0 <= y_mp < height and 0 <= x_mp < width:
            # Apply inverse deformation at this moving point location
            displacement = inverse_deformation[:, 0, y_mp, x_mp]
            fixed_coord = mp + displacement
            fixed_coord_int = np.round(fixed_coord).astype(int)
            transformed_moving_points.append(fixed_coord_int)
            print(f"Moving point {i}: {mp} -> fixed coord {fixed_coord_int} (displacement: {displacement})")
        else:
            print(f"Moving point {i}: {mp} is out of bounds, keeping original")
            transformed_moving_points.append(mp.copy())
    
    transformed_moving_points = np.array(transformed_moving_points)
    
    # Similarly, apply forward deformation to original fixed points
    transformed_fixed_points = []
    for i, fp in enumerate(original_fpoints):
        z_fp, y_fp, x_fp = fp.astype(int)
        
        if 0 <= y_fp < height and 0 <= x_fp < width:
            displacement = forward_deformation[:, 0, y_fp, x_fp]
            moving_coord = fp + displacement
            moving_coord_int = np.round(moving_coord).astype(int)
            transformed_fixed_points.append(moving_coord_int)
            print(f"Fixed point {i}: {fp} -> moving coord {moving_coord_int} (displacement: {displacement})")
        else:
            print(f"Fixed point {i}: {fp} is out of bounds, keeping original")
            transformed_fixed_points.append(fp.copy())
    
    transformed_fixed_points = np.array(transformed_fixed_points)
    
    if plot:
        # Create comprehensive comparison visualization
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        # Row 1: Forward deformation fields
        ax = axes[0, 0]
        im1 = ax.imshow(forward_deformation[2, 0], cmap='RdBu', origin='lower')
        ax.set_title('Forward X-Displacement\n(Fixed→Moving)')
        plt.colorbar(im1, ax=ax, label='X displacement')
        
        ax = axes[0, 1] 
        im2 = ax.imshow(forward_deformation[1, 0], cmap='RdBu', origin='lower')
        ax.set_title('Forward Y-Displacement\n(Fixed→Moving)')
        plt.colorbar(im2, ax=ax, label='Y displacement')
        
        ax = axes[0, 2]
        forward_magnitude = np.sqrt(forward_deformation[1, 0]**2 + forward_deformation[2, 0]**2)
        im3 = ax.imshow(forward_magnitude, cmap='viridis', origin='lower')
        ax.set_title('Forward Displacement\nMagnitude')
        plt.colorbar(im3, ax=ax, label='Magnitude')
        
        # Row 2: Inverse deformation fields
        ax = axes[1, 0]
        im4 = ax.imshow(inverse_deformation[2, 0], cmap='RdBu', origin='lower')
        ax.set_title('Inverse X-Displacement\n(Moving→Fixed)')
        plt.colorbar(im4, ax=ax, label='X displacement')
        
        ax = axes[1, 1]
        im5 = ax.imshow(inverse_deformation[1, 0], cmap='RdBu', origin='lower')
        ax.set_title('Inverse Y-Displacement\n(Moving→Fixed)')
        plt.colorbar(im5, ax=ax, label='Y displacement')
        
        ax = axes[1, 2]
        inverse_magnitude = np.sqrt(inverse_deformation[1, 0]**2 + inverse_deformation[2, 0]**2)
        im6 = ax.imshow(inverse_magnitude, cmap='viridis', origin='lower')
        ax.set_title('Inverse Displacement\nMagnitude')
        plt.colorbar(im6, ax=ax, label='Magnitude')
        
        # Row 3: Point correspondences
        ax = axes[2, 0]
        # Original correspondences
        ax.scatter(original_mpoints[:, 2], original_mpoints[:, 1], 
                  c='red', s=80, alpha=0.8, label='Original moving points', 
                  marker='o', edgecolors='black')
        ax.scatter(original_fpoints[:, 2], original_fpoints[:, 1], 
                  c='blue', s=80, alpha=0.8, label='Original fixed points', 
                  marker='s', edgecolors='black')
        
        # Draw original correspondence lines
        for i in range(len(original_mpoints)):
            ax.plot([original_mpoints[i, 2], original_fpoints[i, 2]], 
                   [original_mpoints[i, 1], original_fpoints[i, 1]], 
                   'gray', alpha=0.3, linewidth=1)
        
        ax.set_title('Original Point\nCorrespondences')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        
        ax = axes[2, 1]
        # Moving points transformed to fixed space using inverse deformation
        ax.scatter(original_mpoints[:, 2], original_mpoints[:, 1], 
                  c='red', s=80, alpha=0.6, label='Original moving points', 
                  marker='o', edgecolors='black')
        ax.scatter(transformed_moving_points[:, 2], transformed_moving_points[:, 1], 
                  c='darkred', s=80, alpha=0.8, label='Moving→Fixed (inverse)', 
                  marker='^', edgecolors='black')
        ax.scatter(original_fpoints[:, 2], original_fpoints[:, 1], 
                  c='blue', s=80, alpha=0.8, label='Original fixed points', 
                  marker='s', edgecolors='black')
        
        # Draw transformation lines
        for i in range(len(original_mpoints)):
            ax.plot([original_mpoints[i, 2], transformed_moving_points[i, 2]], 
                   [original_mpoints[i, 1], transformed_moving_points[i, 1]], 
                   'red', alpha=0.5, linewidth=1)
        
        ax.set_title('Moving Points Transformed\nto Fixed Space')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        
        ax = axes[2, 2]
        # Fixed points transformed to moving space using forward deformation
        ax.scatter(original_fpoints[:, 2], original_fpoints[:, 1], 
                  c='blue', s=80, alpha=0.6, label='Original fixed points', 
                  marker='s', edgecolors='black')
        ax.scatter(transformed_fixed_points[:, 2], transformed_fixed_points[:, 1], 
                  c='darkblue', s=80, alpha=0.8, label='Fixed→Moving (forward)', 
                  marker='v', edgecolors='black')
        ax.scatter(original_mpoints[:, 2], original_mpoints[:, 1], 
                  c='red', s=80, alpha=0.8, label='Original moving points', 
                  marker='o', edgecolors='black')
        
        # Draw transformation lines
        for i in range(len(original_fpoints)):
            ax.plot([original_fpoints[i, 2], transformed_fixed_points[i, 2]], 
                   [original_fpoints[i, 1], transformed_fixed_points[i, 1]], 
                   'blue', alpha=0.5, linewidth=1)
        
        ax.set_title('Fixed Points Transformed\nto Moving Space')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Print comparison statistics
        print(f"\nTransformation Statistics:")
        print(f"Number of points processed: {len(original_mpoints)}")
        
        # Calculate how well the transformations match the expected correspondences
        moving_to_fixed_distances = []
        fixed_to_moving_distances = []
        
        for i in range(len(original_mpoints)):
            # Distance between transformed moving point and its corresponding fixed point
            dist_mf = np.sqrt(np.sum((transformed_moving_points[i] - original_fpoints[i])**2))
            moving_to_fixed_distances.append(dist_mf)
            
            # Distance between transformed fixed point and its corresponding moving point
            dist_fm = np.sqrt(np.sum((transformed_fixed_points[i] - original_mpoints[i])**2))
            fixed_to_moving_distances.append(dist_fm)
        
        moving_to_fixed_distances = np.array(moving_to_fixed_distances)
        fixed_to_moving_distances = np.array(fixed_to_moving_distances)
        
        print(f"Moving→Fixed transformation accuracy:")
        print(f"  Mean distance to target: {moving_to_fixed_distances.mean():.3f}")
        print(f"  Max distance to target: {moving_to_fixed_distances.max():.3f}")
        print(f"  Std deviation: {moving_to_fixed_distances.std():.3f}")
        
        print(f"Fixed→Moving transformation accuracy:")
        print(f"  Mean distance to target: {fixed_to_moving_distances.mean():.3f}")
        print(f"  Max distance to target: {fixed_to_moving_distances.max():.3f}")
        print(f"  Std deviation: {fixed_to_moving_distances.std():.3f}")
        
        # Check consistency - if we apply forward then inverse, do we get back close to original?
        consistency_check = []
        for i in range(len(original_fpoints)):
            fp = original_fpoints[i]
            # Apply forward transformation
            if 0 <= fp[1] < height and 0 <= fp[2] < width:
                mp_transformed = fp + forward_deformation[:, 0, fp[1], fp[2]]
                mp_transformed_int = np.round(mp_transformed).astype(int)
                
                # Apply inverse transformation  
                if 0 <= mp_transformed_int[1] < height and 0 <= mp_transformed_int[2] < width:
                    fp_reconstructed = mp_transformed + inverse_deformation[:, 0, mp_transformed_int[1], mp_transformed_int[2]]
                    reconstruction_error = np.sqrt(np.sum((fp - fp_reconstructed)**2))
                    consistency_check.append(reconstruction_error)
        
        if consistency_check:
            consistency_check = np.array(consistency_check)
            print(f"Round-trip consistency (Fixed→Moving→Fixed):")
            print(f"  Mean reconstruction error: {consistency_check.mean():.3f}")
            print(f"  Max reconstruction error: {consistency_check.max():.3f}")
    
    return forward_coords_int, inverse_coords_int, inverse_data_obj, transformed_moving_points, transformed_fixed_points


def cluster_by_movement_direction(moving_points, fixed_points, method='displacement_direction', 
                                  plot=True, title="Direction-Based Clustering"):
    """
    Cluster double-helix points based on their movement direction.
    
    Parameters:
    -----------
    moving_points : numpy.ndarray
        Array of moving points, shape (N, 3) with (z, y, x) coordinates
    fixed_points : numpy.ndarray
        Array of fixed points, shape (N, 3) with (z, y, x) coordinates  
    method : str, default='displacement_direction'
        Clustering method:
        - 'displacement_direction': Cluster by displacement vector direction
        - 'y_direction': Cluster by whether points move up or down in Y
        - 'quadrant': Cluster by displacement vector quadrant
        - 'helix_phase': Cluster by estimated phase in helix pattern
    plot : bool, default=True
        Whether to create visualization plots
    title : str
        Title for the plots
        
    Returns:
    --------
    clusters : list of lists
        Each sublist contains indices of points belonging to the same cluster
    cluster_labels : numpy.ndarray
        Array of cluster labels for each point
    displacement_vectors : numpy.ndarray
        Displacement vectors (fixed - moving) for each point
    """
    
    # Calculate displacement vectors (from moving to fixed)
    displacement_vectors = fixed_points - moving_points
    
    print(f"Analyzing {len(moving_points)} point pairs for directional clustering...")
    print(f"Clustering method: {method}")
    
    # Different clustering methods
    if method == 'displacement_direction':
        # Cluster based on the angle of displacement in XY plane
        angles = np.arctan2(displacement_vectors[:, 1], displacement_vectors[:, 2])  # Y, X displacement
        angles_deg = np.degrees(angles)
        
        # Normalize angles to [0, 360)
        angles_deg = (angles_deg + 360) % 360
        
        # Cluster into 4 quadrants or more sophisticated directional groups
        cluster_labels = np.zeros(len(angles_deg), dtype=int)
        
        # Define directional sectors (can be adjusted)
        n_sectors = 4  # Number of directional sectors
        sector_size = 360 / n_sectors
        
        for i, angle in enumerate(angles_deg):
            cluster_labels[i] = int(angle // sector_size)
        
        cluster_info = f"Directional sectors (n={n_sectors})"
        
    elif method == 'y_direction':
        # Simple clustering: up vs down movement in Y direction
        cluster_labels = (displacement_vectors[:, 1] >= 0).astype(int)  # 0 for down, 1 for up
        cluster_info = "Y-direction (0=down, 1=up)"
        
    elif method == 'x_direction':
        # Simple clustering: up vs down movement in x direction
        cluster_labels = (displacement_vectors[:, 2] >= 0).astype(int)  # 0 for down, 1 for up
        cluster_info = "X-direction (0=right, 1=left)"
        
    elif method == 'quadrant':
        # Cluster based on XY displacement quadrant
        x_disp = displacement_vectors[:, 2]  # X displacement
        y_disp = displacement_vectors[:, 1]  # Y displacement
        
        cluster_labels = np.zeros(len(x_disp), dtype=int)
        
        # Quadrant assignment: 0=+X+Y, 1=-X+Y, 2=-X-Y, 3=+X-Y
        for i in range(len(x_disp)):
            if x_disp[i] >= 0 and y_disp[i] >= 0:
                cluster_labels[i] = 0  # +X, +Y
            elif x_disp[i] < 0 and y_disp[i] >= 0:
                cluster_labels[i] = 1  # -X, +Y
            elif x_disp[i] < 0 and y_disp[i] < 0:
                cluster_labels[i] = 2  # -X, -Y
            else:  # x_disp[i] >= 0 and y_disp[i] < 0
                cluster_labels[i] = 3  # +X, -Y
                
        cluster_info = "Quadrants (0=+X+Y, 1=-X+Y, 2=-X-Y, 3=+X-Y)"
        
    elif method == 'helix_phase':
        # Estimate helix phase based on X position and displacement pattern
        x_positions = moving_points[:, 2]  # X coordinates of moving points
        
        # Estimate local derivative/slope of the helix at each point
        # For a double helix, points moving in opposite Y directions should be in different clusters
        y_displacement = displacement_vectors[:, 1]
        
        # Simple phase estimation: alternate clusters based on X position segments
        x_min, x_max = x_positions.min(), x_positions.max()
        x_normalized = (x_positions - x_min) / (x_max - x_min)  # Normalize to [0, 1]
        
        # Estimate number of cycles in the helix and assign phase-based clusters
        estimated_cycles = 3  # This could be estimated from the data
        phase_factor = estimated_cycles * 2 * np.pi
        estimated_phases = x_normalized * phase_factor
        
        # Cluster based on sine of phase - positive vs negative regions
        cluster_labels = (np.sin(estimated_phases) >= 0).astype(int)
        
        cluster_info = f"Helix phase estimation ({estimated_cycles} cycles)"
        
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Convert labels to list of index lists
    unique_labels = np.unique(cluster_labels)
    clusters = []
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels == label)[0].tolist()
        clusters.append(cluster_indices)
    
    # Print cluster statistics
    print(f"\nClustering Results ({cluster_info}):")
    print(f"Number of clusters: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: {len(cluster)} points (indices: {cluster[:5]}{'...' if len(cluster) > 5 else ''})")
        
        # Calculate cluster statistics
        cluster_displacements = displacement_vectors[cluster]
        mean_displacement = np.mean(cluster_displacements, axis=0)
        print(f"  Mean displacement: Z={mean_displacement[0]:.3f}, Y={mean_displacement[1]:.3f}, X={mean_displacement[2]:.3f}")
        
        if len(cluster_displacements) > 1:
            std_displacement = np.std(cluster_displacements, axis=0)
            print(f"  Std displacement:  Z={std_displacement[0]:.3f}, Y={std_displacement[1]:.3f}, X={std_displacement[2]:.3f}")
    
    if plot:
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Plot 1: Original point correspondences colored by cluster
        ax = axes[0, 0]
        colors = plt.cm.Set1(np.linspace(0, 1, len(clusters)))
        
        for i, (cluster, color) in enumerate(zip(clusters, colors)):
            cluster_moving = moving_points[cluster]
            cluster_fixed = fixed_points[cluster]
            
            ax.scatter(cluster_moving[:, 2], cluster_moving[:, 1], 
                      c=[color], s=60, alpha=0.8, label=f'Cluster {i} Moving', 
                      marker='o', edgecolors='black', linewidth=0.5)
            ax.scatter(cluster_fixed[:, 2], cluster_fixed[:, 1], 
                      c=[color], s=60, alpha=0.8, 
                      marker='s', edgecolors='black', linewidth=0.5)
            
            # Draw correspondence lines
            for j in range(len(cluster_moving)):
                ax.plot([cluster_moving[j, 2], cluster_fixed[j, 2]], 
                       [cluster_moving[j, 1], cluster_fixed[j, 1]], 
                       color=color, alpha=0.3, linewidth=1)
        
        ax.set_title('Clustered Point Correspondences\n(Circles=Moving, Squares=Fixed)')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Displacement vectors colored by cluster
        ax = axes[0, 1]
        for i, (cluster, color) in enumerate(zip(clusters, colors)):
            cluster_displacements = displacement_vectors[cluster]
            ax.scatter(cluster_displacements[:, 2], cluster_displacements[:, 1], 
                      c=[color], s=60, alpha=0.8, label=f'Cluster {i}', 
                      marker='o', edgecolors='black', linewidth=0.5)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Displacement Vectors by Cluster')
        ax.set_xlabel('X displacement')
        ax.set_ylabel('Y displacement')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Displacement magnitude vs X position
        ax = axes[0, 2]
        displacement_magnitudes = np.sqrt(np.sum(displacement_vectors**2, axis=1))
        
        for i, (cluster, color) in enumerate(zip(clusters, colors)):
            cluster_x = moving_points[cluster, 2]
            cluster_magnitudes = displacement_magnitudes[cluster]
            ax.scatter(cluster_x, cluster_magnitudes, 
                      c=[color], s=60, alpha=0.8, label=f'Cluster {i}', 
                      marker='o', edgecolors='black', linewidth=0.5)
        
        ax.set_title('Displacement Magnitude vs X Position')
        ax.set_xlabel('X coordinate (moving points)')
        ax.set_ylabel('Displacement magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Direction analysis
        ax = axes[1, 0]
        if method == 'displacement_direction':
            angles = np.arctan2(displacement_vectors[:, 1], displacement_vectors[:, 2])
            angles_deg = (np.degrees(angles) + 360) % 360
            
            for i, (cluster, color) in enumerate(zip(clusters, colors)):
                cluster_angles = angles_deg[cluster]
                ax.scatter(cluster_angles, np.ones(len(cluster_angles)) * i, 
                          c=[color], s=60, alpha=0.8, label=f'Cluster {i}', 
                          marker='o', edgecolors='black', linewidth=0.5)
            
            ax.set_title('Displacement Direction Angles by Cluster')
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Cluster')
            ax.set_xlim(0, 360)
            ax.grid(True, alpha=0.3)
            
        elif method == 'y_direction':
            y_displacements = displacement_vectors[:, 1]
            for i, (cluster, color) in enumerate(zip(clusters, colors)):
                cluster_y_disp = y_displacements[cluster]
                ax.hist(cluster_y_disp, bins=20, alpha=0.6, color=color, 
                       label=f'Cluster {i}', edgecolor='black')
            
            ax.set_title('Y Displacement Distribution by Cluster')
            ax.set_xlabel('Y displacement')
            ax.set_ylabel('Count')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero displacement')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Cluster separation in 3D displacement space
        ax = axes[1, 1]
        # Project to 2D for visualization (Y vs X displacement)
        for i, (cluster, color) in enumerate(zip(clusters, colors)):
            cluster_displacements = displacement_vectors[cluster]
            ax.scatter(cluster_displacements[:, 2], cluster_displacements[:, 1], 
                      c=[color], s=80, alpha=0.8, label=f'Cluster {i}', 
                      marker=['o', 's', '^', 'v', 'D'][i % 5], edgecolors='black', linewidth=1)
        
        # Add quadrant lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Cluster Separation in Displacement Space')
        ax.set_xlabel('X displacement')
        ax.set_ylabel('Y displacement')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Position along helix vs cluster assignment
        ax = axes[1, 2]
        x_positions = moving_points[:, 2]
        y_positions = moving_points[:, 1]
        
        for i, (cluster, color) in enumerate(zip(clusters, colors)):
            cluster_x = x_positions[cluster]
            cluster_y = y_positions[cluster]
            ax.scatter(cluster_x, cluster_y, 
                      c=[color], s=60, alpha=0.8, label=f'Cluster {i}', 
                      marker='o', edgecolors='black', linewidth=0.5)
        
        ax.set_title('Original Helix Pattern\nColored by Cluster')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{title}\nMethod: {cluster_info}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    return clusters, cluster_labels, displacement_vectors


def get_cluster_indices(moving_points, fixed_points, method='y_direction', verbose=False):
    """
    Helper function to quickly get cluster indices without visualization.
    
    Parameters:
    -----------
    moving_points : numpy.ndarray
        Array of moving points, shape (N, 3) with (z, y, x) coordinates
    fixed_points : numpy.ndarray
        Array of fixed points, shape (N, 3) with (z, y, x) coordinates  
    method : str, default='y_direction'
        Clustering method to use
    verbose : bool, default=False
        Whether to print cluster statistics
        
    Returns:
    --------
    cluster_indices : list of numpy arrays
        Each array contains the indices for points in that cluster
    cluster_info : dict
        Information about each cluster including indices and statistics
    """
    
    # Run clustering without plots
    clusters, labels, displacements = cluster_by_movement_direction(
        moving_points, fixed_points, method=method, plot=False, 
        title="Clustering for Index Extraction"
    )
    
    # Convert to numpy arrays for easier indexing
    cluster_indices = [np.array(cluster, dtype=int) for cluster in clusters]
    
    # Create detailed cluster information
    cluster_info = {}
    for i, indices in enumerate(cluster_indices):
        cluster_moving = moving_points[indices]
        cluster_fixed = fixed_points[indices]
        cluster_displacements = displacements[indices]
        
        cluster_info[f'cluster_{i}'] = {
            'indices': indices,
            'size': len(indices),
            'moving_points': cluster_moving,
            'fixed_points': cluster_fixed,
            'displacements': cluster_displacements,
            'mean_displacement': np.mean(cluster_displacements, axis=0),
            'std_displacement': np.std(cluster_displacements, axis=0) if len(indices) > 1 else np.zeros(3),
            'displacement_magnitude': np.sqrt(np.sum(cluster_displacements**2, axis=1)),
        }
    
    if verbose:
        print(f"Extracted {len(cluster_indices)} clusters using {method} method:")
        for i, indices in enumerate(cluster_indices):
            info = cluster_info[f'cluster_{i}']
            print(f"  Cluster {i}: {len(indices)} points")
            print(f"    Indices: {indices[:10].tolist()}{'...' if len(indices) > 10 else ''}")
            print(f"    Mean displacement: Z={info['mean_displacement'][0]:.2f}, Y={info['mean_displacement'][1]:.2f}, X={info['mean_displacement'][2]:.2f}")
            print(f"    Mean magnitude: {info['displacement_magnitude'].mean():.2f}")
    
    return cluster_indices, cluster_info


def compose_deformation_fields(def1, def2):
    """
    Compose two deformation fields: def1 followed by def2.
    def1, def2: shape (channels, batch, H, W)
    Returns: composed deformation field, same shape
    """
    _, batch, H, W = def1.shape
    composed = np.zeros_like(def1)
    # Create meshgrid of coordinates
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    for b in range(batch):
        # Get where each pixel moves after def1, y1 and x1 are the new coordinates after applying def1
        y1 = yy + def1[1, b]
        x1 = xx + def1[2, b]
        # Sample def2 at the new locations (use nearest neighbor for simplicity)
        y1_clipped = np.clip(np.round(y1).astype(int), 0, H-1)
        x1_clipped = np.clip(np.round(x1).astype(int), 0, W-1)
        # Compose: total displacement = def1 + def2 at warped location
        # Fancy indexing: For each pixel (i, j), it fetches the value at [1, b, y1_clipped[i, j], x1_clipped[i, j]]
        composed[1, b] = def1[1, b] + def2[1, b, y1_clipped, x1_clipped]
        composed[2, b] = def1[2, b] + def2[2, b, y1_clipped, x1_clipped]
    return composed

