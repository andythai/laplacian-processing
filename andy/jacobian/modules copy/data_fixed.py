import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from modules import jacobian, laplacian_fixed
import networkx as nx

def points_in_grid(points, center, grid_size=7):
    """
    Get the indexes of the points that lie within a grid centered at the center point.
    
    Args:
    - points (np.array): The points to filter
    - center (tuple): The center of the grid
    - grid_size (int): The size of the grid
    
    Returns:
    - np.array: The indexes of the points that lie within the grid
    """
    cz, cy, cx = center
    radius = grid_size // 2  # 3 for a 7x7 grid
    
    # Filtering points that lie within the 7x7 boundary
    filtered_indexes = [
        i for i, (z, y, x) in enumerate(points)
        if (cx - radius) <= x <= (cx + radius) and (cy - radius) <= y <= (cy + radius) and not np.array_equal(center, [z, y, x])
    ]
    
    return np.array(filtered_indexes)

class Data:
    def __init__(self, mpoints: np.ndarray, fpoints: np.ndarray, resolution: np.ndarray):
        """Initialize the Data object with matched points and resolution.
        
        Args:
            mpoints (np.ndarray): Matched points in the model image.
            fpoints (np.ndarray): Corresponding points in the frame image.
            resolution (np.ndarray): Resolution of the image (y, x)
        """
        self.mpoints = mpoints
        self.fpoints = fpoints
        self.resolution = resolution
        self.deformation, self.jdet_field = self._create_grid()
        
        self.A = None  # Placeholder for the Laplacian matrix
        self.Xd = None  # Placeholder for the X deformation field
        self.Yd = None  # Placeholder for the Y deformation field
        self.Zd = None  # Placeholder for the Z deformation field

        
    def _create_grid(self):
        """Create a grid of correspondences between matched points.

        Args:
            mpoints (np.ndarray): Matched points in the model image.
            fpoints (np.ndarray): Corresponding points in the frame image.
            
        Returns:
            np.ndarray: Deformation field.
            np.ndarray: Jacobian determinant field.
        """
        # Run Laplacian and Jacobian determinant calculations and save the Jacobian determinant field
        grid_resolution = np.zeros((1, self.resolution[0], self.resolution[1]))
        deformation, A, Xd, Yd, Zd = laplacian_fixed.sliceToSlice3DLaplacian(grid_resolution, self.mpoints, self.fpoints)
        jdet_field = jacobian.sitk_jacobian_determinant(deformation)[0]
        self.A = A
        self.Xd = Xd
        self.Yd = Yd
        self.Zd = Zd
        return deformation, jdet_field
    
    def get_error(self):
        """ Calculate the error in the deformation field."""
        # L2 error calculation
        deformation_z = self.deformation[0, 0, :, :]  # Get the first slice (only slice)
        deformation_y = self.deformation[1, 0, :, :]  # Get the second slice
        deformation_x = self.deformation[2, 0, :, :]  # Get the third slice
        # Reshape
        
        # Calculate the error
        err_z = self.A @ deformation_z.flatten() - self.Zd
        err_y = self.A @ deformation_y.flatten() - self.Yd
        err_x = self.A @ deformation_x.flatten() - self.Xd
        return err_z + err_y + err_x
        
        
    def compare(self, idx1: int, idx2: int, grid_size=None, title=None, figsize=(10, 5), fontsize=6, show=True):
        """Compare two correspondences in the grid.
        
        Args:
            idx1 (int): Index of the first point.
            idx2 (int): Index of the second point.
            
        Returns:
            Data: New Data object with only the two correspondences compared.
        """
        mpoint1 = self.mpoints[idx1].copy()
        fpoint1 = self.fpoints[idx1].copy()
        mpoint2 = self.mpoints[idx2].copy()
        fpoint2 = self.fpoints[idx2].copy()
        
        if grid_size is not None:
            center = grid_size // 2
            center_array = np.array([0, center, center])
            #center_diff = center_array - mpoint1
            center_diff = center_array - fpoint1
            
            # Move points to center of grid
            mpoint1 += center_diff
            fpoint1 += center_diff
            mpoint2 += center_diff
            fpoint2 += center_diff
            #print("Center array:", center_array)
            #print("Center diff:", center_diff)
            #print("Moved mpoint1:", mpoint1)
            #print("Moved fpoint1:", fpoint1)
            #print("Moved mpoint2:", mpoint2)
            #print("Moved fpoint2:", fpoint2)

        mpoints = np.vstack((mpoint1, mpoint2))
        fpoints = np.vstack((fpoint1, fpoint2))
        
        if grid_size is None:
            d = Data(mpoints, fpoints, self.resolution)
        else:
            d = Data(mpoints, fpoints, (grid_size, grid_size))
            
        if show:
            if title is None:
                title = f"Comparing {idx1} and {idx2}, min = {d.min():.2f}"
            d.show(title=title, figsize=figsize, fontsize=fontsize)
        return d
    
    
    def min(self):
        """Find the minimum Jacobian determinant value in the grid.
        
        Returns:
            float: Minimum Jacobian determinant value.
        """
        return np.min(self.jdet_field)
    
    
    def count_negatives(self):
        """
        Counts the number of negative Jacobian determinants in the grid.
        
        Returns:
            int: Number of negative Jacobian determinants.
        """
        return np.sum(self.jdet_field <= 0)
    
    
    def get_negative_coordinates(self, show=True):
        """
        Get the coordinates of negative Jacobian determinants in the grid.
        
        Args:
            show (bool): Whether to print the coordinates of the negative Jacobian determinants.
            
        Returns:
            np.ndarray: Array of coordinates of negative Jacobian determinants.
        """
        # Get coordinate of negative Jacobian determinants
        negatives = np.argwhere(self.jdet_field <= 0)
        if show:
            print("Negative Jacobian determinants:")
            for y, x in negatives:
                print(f"({y}, {x})")
        return negatives
    
    
    def build_conflict_graph(self, threshold: float = 0.0, grid_size=101, show_compare=False):
        """Build a conflict graph based on the Jacobian determinant field.
        
        Args:
            threshold (float, optional): Threshold for conflict detection. Defaults to 0.0.
            
        Returns:
            nx.Graph: Conflict graph where keys are indices of moving points and values are lists of conflicting indices.
        """
        conflict_graph = nx.Graph()
        scores = np.zeros(self.mpoints.shape[0])
        
        #scores = np.full(self.mpoints.shape[0], np.inf)
        
        # Check for conflicts between correspondence pairs
        num_correspondences = self.mpoints.shape[0]
        #for i in tqdm(range(num_correspondences)):
        for i in range(num_correspondences):
            
            # Filter points and compare only to nearby points
            current_point = self.fpoints[i]
            neighbor_indices = points_in_grid(self.fpoints, current_point, grid_size=grid_size)
            neighbor_indices = neighbor_indices[neighbor_indices >= i]
            
            #for j in tqdm(range(i + 1, num_correspondences), leave=False):
            #for nj in tqdm(range(len(neighbor_indices)), leave=False):
            for nj in range(len(neighbor_indices)):
                j = neighbor_indices[nj]
                d = self.compare(i, j, grid_size=grid_size, show=show_compare)
                min_jdet = d.min()
                scores[i] += min_jdet
                scores[j] += min_jdet
                if min_jdet < threshold:
                    conflict_graph.add_edge(i, j, weight=min_jdet)
        return conflict_graph, scores
        
        
    def remove(self, idx: int, title=None, show=True, figsize=(10, 5), fontsize=6):
        """Remove a correspondence by index.
        
        Args:
            idx (int): Index of the correspondence to remove.
            
        Returns:
            Data: New Data object with the specified correspondence removed.
        """
        mpoints = np.delete(self.mpoints, idx, axis=0)
        fpoints = np.delete(self.fpoints, idx, axis=0)
        d = Data(mpoints, fpoints, self.resolution)
        if show:
            if title is None:
                title = f"Removing {idx}, min = {d.min():.2f}"
            d.show(title=title, figsize=figsize, fontsize=fontsize)
        return d
        
        
    def show(self, title: str = None, figsize: tuple = (10, 5), fontsize: int = 6, show_text=True, show_axis=True, show_correspondence=True, show_orientation=False, binarize_negatives=False):
        """Show the deformation field and the Jacobian determinant field.
        
        Args:
            title (str, optional): Title of the plot. Defaults to None.
            figsize (tuple, optional): Size of the plot. Defaults to (10, 5).
            fontsize (int, optional): Font size of the text. Defaults to 6.
            show_text (bool, optional): Whether to show text on the plot. Defaults to True.
            show_axis (bool, optional): Whether to show the normalized axis text or use the default. Defaults to True.
        """
        norm = mcolors.TwoSlopeNorm(vmin=min(self.jdet_field.min(), -1), vcenter=0, vmax=self.jdet_field.max())
        #norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        
        jdet_field = self.jdet_field.copy()
        
        if binarize_negatives:
            jdet_field[jdet_field < 0] = -1
            jdet_field[jdet_field > 0] = 1  # Clip values to [-1, 1] for better visualization

        f = plt.figure(figsize=figsize)
        plt.imshow(jdet_field, cmap="seismic", norm=norm)
        plt.colorbar()
        
        # Draw correspondence arrows
        if show_correspondence:
            for i in range(self.mpoints.shape[0]):
                plt.arrow(self.mpoints[i, 2], self.mpoints[i, 1], self.fpoints[i, 2] - self.mpoints[i, 2], self.fpoints[i, 1] - self.mpoints[i, 1],
                        head_width=0.15, head_length=0.15, fc='g', ec='g')
            
            # Draw index at each moving point
            #if fontsize != 0:
            for i in range(self.mpoints.shape[0]):
                plt.text(self.mpoints[i, 2], self.mpoints[i, 1] + 0.4, str(i), color='green', ha='center', va='bottom', fontsize=12, weight='bold')
            
        # Plot the coordinate text
        if show_text:
            for y in range(self.jdet_field.shape[0]):
                for x in range(self.jdet_field.shape[1]):
                    # Get the pixel information                
                    curr_displacement = self.deformation[:, 0, y, x]  # Get first slice (only slice)
                    displacement_y = curr_displacement[1]
                    displacement_x = curr_displacement[2]

                    # Get the angle information
                    curr_theta_rad = np.arctan2(displacement_y, displacement_x)
                    if curr_theta_rad < 0:  # Ensure the angle is in the range [0, 2pi)
                        curr_theta_rad += 2 * np.pi
                    curr_theta = np.abs(np.degrees(curr_theta_rad))
                    
                    # Get the magnitude information
                    curr_magnitude = np.linalg.norm(curr_displacement)
                    jdet = self.jdet_field[y, x]
                    
                    # Setup text to display
                    coord_text = f"({displacement_y:.2f}, {displacement_x:.2f})\n"
                    jdet_text = f"{jdet:.2f} J"
                    vector_text = f"\n{curr_theta:.2f}°\n" + f"∥{curr_magnitude:.2f}∥"
                    #pixel_text = coord_text + jdet_text + vector_text
                    pixel_text = coord_text + jdet_text
                    if fontsize != 0:
                        plt.text(x, y, pixel_text, color='black', ha='center', va='center', fontsize=fontsize, weight='normal')
                                    
                    # Draw the vector direction
                    if show_orientation:
                        magnitude_scale = 0.33  # To scale the vector magnitude for visualization purposes
                        plt.arrow(x, y, np.cos(curr_theta_rad) * magnitude_scale, np.sin(curr_theta_rad) * magnitude_scale, head_width=0.15, head_length=0.15, fc='black', ec='black')            

        if title is None:
            title = f"{self.resolution} Jacobian determinants"
        plt.title(title)
        
        if show_axis and fontsize != 0:
            offset_y = 0 if self.jdet_field.shape[0] % 2 == 0 else 1
            offset_x = 0 if self.jdet_field.shape[1] % 2 == 0 else 1
            plt.yticks(np.arange(self.jdet_field.shape[0]), labels=np.arange(-(self.jdet_field.shape[0] // 2) - offset_y, self.jdet_field.shape[0] // 2) + offset_y)
            plt.xticks(np.arange(self.jdet_field.shape[1]), labels=np.arange(-(self.jdet_field.shape[1] // 2) - offset_x, self.jdet_field.shape[1] // 2) + offset_x)
        plt.show()
        f.clear()
        plt.close(f)
        
        
