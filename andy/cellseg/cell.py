# Description: This file contains the implementation of the Cell class, which represents a cell in a 2D image.
# Author: Andy Thai

# Standard imports
import pickle
import math

# Third-party imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from skimage.draw import polygon
from skimage.measure import find_contours
#from scipy.sparse import coo_array
import sparse

# TODO: Compute tubularity and major axis / minor axis ratio
# TODO: Do something that uses local peak finding to find the center of the cell too

def compute_dice_coefficient(binary_image1: np.ndarray, binary_image2: np.ndarray) -> float:
    """
    Compute the Dice coefficient between two binary images.
    
    Parameters:
    - binary_image1: 2D numpy array representing the first binary image.
    - binary_image2: 2D numpy array representing the second binary image.
    
    Returns:
    - dice_coefficient: The Dice coefficient between the two binary images.
    """
    intersection = np.logical_and(binary_image1, binary_image2)
    return 2. * intersection.sum() / (binary_image1.sum() + binary_image2.sum())


class Cell:
    def __init__(self, image: np.ndarray, z: int=None, index: int=None):
        """
        Initialize the cell object.
        
        Parameters:
        - image: 2D numpy array representing the binary image of the cell.
        - z: Z-coordinate of the cell.
        - index: Index of the cell.
        """
        self.image = sparse.COO.from_numpy(image.astype(bool))  # Use sparse to save on memory
        self.z = z
        self.index = index
        
    
    def get_area(self) -> int:
        """
        Compute the area of the cell image.
        
        Returns:
        - area: Area of the cell.
        """
        return np.sum(self.image)
    
    
    def get_centroid(self) -> tuple:
        """
        Compute the centroid of the cell image.
        
        Returns:
        - centroid: Tuple representing the (row, column) coordinates of the centroid.
        """
        # Find the coordinates of the foreground pixels
        rows, cols = np.nonzero(self.image)
        
        # Compute the centroid
        centroid_row = np.mean(rows)
        centroid_col = np.mean(cols)
        
        return (centroid_row, centroid_col)
    
    
    def get_medoid(self) -> tuple:
        """
        Compute the medoid of the cell image.
        
        Returns:
        - medoid: Tuple representing the (row, column) coordinates of the medoid.
        """
        # Find the coordinates of the foreground pixels
        points = np.column_stack(np.nonzero(self.image))
        
        # Compute the pairwise distances
        distances = cdist(points, points)
        
        # Compute the sum of distances for each point
        sum_distances = np.sum(distances, axis=1)
        
        # Find the index of the point with the minimum sum of distances
        medoid_index = np.argmin(sum_distances)
        
        # Get the coordinates of the medoid
        medoid = points[medoid_index]
        
        return (medoid[0], medoid[1])
    
    
    def get_convexity(self) -> float:
        """
        Compute the convexity of the cell
        
        Returns:
        - convexity: Convexity of the cell
        """
        convex_hull_image = self.get_convex_hull_image()
        convex_hull_area = np.sum(convex_hull_image)
        if convex_hull_area == 0:
            return 0
        else:
            return self.get_area() / np.sum(convex_hull_image)
    

    def get_boundary_points(self) -> np.ndarray:
        """
        Get the boundary points of the binary image.
        
        Returns:
        - boundary_points: Coordinates of the boundary points.
        """
        contours = find_contours(self.image.todense(), level=0.5)
        if len(contours) == 0:
            raise ValueError("No contours found in the binary image")
        return contours[0]
        
        
    def get_convex_hull(self) -> np.ndarray:
        """
        Get the points on the convex hull of a binary image.
        
        Returns:
        - hull_points: Coordinates of the points on the convex hull.
        """
        # Extract the coordinates of the foreground pixels
        points = np.column_stack(np.nonzero(self.image))
        
        # Compute the convex hull
        try:
            hull = ConvexHull(points)
            # Get the coordinates of the points on the convex hull
            hull_points = points[hull.vertices]
        except Exception as e:
            print("Convex hull computation failed at " + str(self) + f". Cell area ({self.get_area()}) is likely too small.")
            hull_points = []

        return hull_points
    
    
    def get_convex_hull_image(self) -> np.ndarray:
        """
        Get the convex hull image of the cell
        
        Returns:
        - convex_hull_image: Binary image of the convex hull.
        """
        convex_hull_points = self.get_convex_hull()
        
        # Create a binary image
        convex_hull_image = np.zeros_like(self.image.todense(), dtype=bool)

        if len(convex_hull_points) != 0:
            # Get the row and column coordinates of the convex hull points
            rr, cc = polygon(convex_hull_points[:, 0], convex_hull_points[:, 1], convex_hull_image.shape)

            # Fill the convex hull area in the binary image
            convex_hull_image[rr, cc] = 1
            
        return convex_hull_image
    
    
    def get_limits(self, spacing: float=3) -> tuple:
        """
        Get the limits of the cell image for plotting.
        
        Parameters:
        - spacing: Spacing around the convex hull.
        
        Returns:
        - limits: Tuple representing the limits of the cell image.
        """
        # Get the limits for the plot
        convex_hull_points = self.get_convex_hull()
        min_y = np.min(convex_hull_points[:, 0])
        max_y = np.max(convex_hull_points[:, 0])
        min_x = np.min(convex_hull_points[:, 1])
        max_x = np.max(convex_hull_points[:, 1])
        x_spacing = int((max_x - min_x) * spacing)
        y_spacing = int((max_y - min_y) * spacing)
        x_hull_lim = (min_x - x_spacing, max_x + x_spacing)
        y_hull_lim = (max_y + y_spacing, min_y - y_spacing)
        return y_hull_lim, x_hull_lim
    
    
    def get_stats(self, section_img: np.ndarray=None, show: bool=True) -> list:
        """
        Get the statistics of the cell.
        
        Parameters:
        - section_img: 2D numpy array representing the section image.
        - show: Whether to show the statistics.
        
        Returns:
        - stats: List of statistics of the cell.
        """
        cell_area = self.get_area()
        convexity = self.get_convexity()
        eccentricity = self.get_eccentricity()
        ellipse_dice = self.get_ellipse_dice()
        centroid = self.get_centroid()
        medoid = self.get_medoid()
        image_shape = self.image.shape
        shape_stats = [cell_area, convexity, eccentricity, ellipse_dice]
        image_stats = [image_shape, centroid, medoid]
        intensity_stats = []
        # Mask the section image with the cell binary image
        if section_img is not None:
            intensity_values = section_img[self.image.todense()]
            img_min = np.min(intensity_values)
            img_max = np.max(intensity_values)
            img_median = np.median(intensity_values)
            img_mean = np.mean(intensity_values)
            img_std = np.std(intensity_values)
            intensity_stats = [img_min, img_max, img_median, img_mean, img_std]
        if show:
            title_prefix = " " * 2
            prefix = " " * 6
            print(str(self))
            print(f"{title_prefix}Geometric stats")
            print(f"{prefix}Cell area:", cell_area, "pixels")
            print(f"{prefix}Convexity:", convexity)
            print(f"{prefix}Eccentricity:", eccentricity)
            print(f"{prefix}Ellipse Dice coefficient:", ellipse_dice)
            print(f"{title_prefix}Image stats")
            print(f"{prefix}Image shape:", image_shape)
            print(f"{prefix}Centroid:", centroid)
            print(f"{prefix}Medoid:", medoid)
            if section_img is not None:
                print(f"{title_prefix}Intensity stats")
                print(f"{prefix}Min:", img_min)
                print(f"{prefix}Max:", img_max)
                print(f"{prefix}Median:", img_median)
                print(f"{prefix}Mean:", img_mean)
                print(f"{prefix}Std:", img_std)
            print()
        return shape_stats + image_stats + intensity_stats
    
    
    def locate(self, section_img: np.ndarray=None, spacing: float=3):
        """
        Shows the location of the cell in the image section.
        
        Parameters:
        - section: 2D numpy array representing the section image.
        - spacing: Spacing around the convex hull.
        """
        medoid = self.get_medoid()
        text_offset = np.max(self.image.shape) * 0.01
        
        # Get the limits for the plot
        y_hull_lim, x_hull_lim = self.get_limits(spacing=spacing)

        
        # Plot the cell and section
        img_todense = self.image.todense()
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        if section_img is not None:
            plt.imshow(section_img, cmap='gray')
        plt.imshow(img_todense, cmap='gray', alpha=0.5)
        plt.plot(medoid[1], medoid[0], 'co', markersize=2)
        plt.text(medoid[1] + text_offset, medoid[0] - text_offset, f"{medoid}", fontsize=6, color='cyan')
        plt.title(str(self))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.subplot(1, 2, 2)
        if section_img is not None:
            plt.imshow(section_img, cmap='gray')
        plt.contour(img_todense, levels=[0.5], colors='red', alpha=0.35)
        plt.plot(medoid[1], medoid[0], 'co', markersize=1)
        plt.xlim(x_hull_lim)
        plt.ylim(y_hull_lim)
        plt.title("Zoomed in view")
        plt.show()
    
    
    def show(self, section_img: np.ndarray=None):
        """
        Show the cell image along with the convex hull and fitted ellipse.
        
        Parameters:
        - section_img: 2D numpy array representing the section image.
        """
        # Get the convex hull points and image
        convex_hull_points = self.get_convex_hull()
        convex_hull_points = np.vstack([convex_hull_points, convex_hull_points[0]])  # Append the first point to close the loop
        convex_hull_image = self.get_convex_hull_image()
        centroid = self.get_centroid()
        medoid = self.get_medoid()
        ellipse_image = self.get_ellipse_image()
        
        # Get the limits for the plot
        min_y = np.min(convex_hull_points[:, 0])
        max_y = np.max(convex_hull_points[:, 0])
        min_x = np.min(convex_hull_points[:, 1])
        max_x = np.max(convex_hull_points[:, 1])
        x_spacing = int((max_x - min_x) * 0.2)
        y_spacing = int((max_y - min_y) * 0.2)
        x_hull_lim = (min_x - x_spacing, max_x + x_spacing)
        y_hull_lim = (max_y + y_spacing, min_y - y_spacing)
        text_offset = np.max([max_x - min_x, max_y - min_y]) * 0.01
        
        # Plot the cell and convex hull
        img_todense = self.image.todense()
        plt.figure(figsize=(10, 5))
        if section_img is not None:
            plt.subplot(1, 2, 1)
        plt.imshow(img_todense, cmap='gray', alpha=1)  # Plot the cell binary image
        plt.imshow(convex_hull_image, cmap=ListedColormap(['black', 'red']), alpha=0.25)  # Plot the convex hull fill
        plt.plot(convex_hull_points[:, 1], convex_hull_points[:, 0], 'r--', lw=1, alpha=0.5, label='Convex hull')  # Dashed lines for convex hull
        plt.plot(convex_hull_points[:, 1], convex_hull_points[:, 0], 'ro', markersize=3)  # Plot the convex hull points
        plt.plot(centroid[1], centroid[0], 'go', markersize=5, label='Centroid')  # Plot the centroid
        plt.text(centroid[1] + text_offset, centroid[0] - text_offset, f"{math.floor(centroid[0]), math.floor(centroid[1])}", fontsize=8, color='green', weight='bold')
        plt.plot(medoid[1], medoid[0], 'co', markersize=5, label='Medoid')  # Plot the centroid
        plt.text(medoid[1] + text_offset, medoid[0] - text_offset, f"{math.floor(medoid[0]), math.floor(medoid[1])}", fontsize=8, color='cyan', weight='bold')
        
        # Create a custom legend entry for the ellipse
        handles, labels = plt.gca().get_legend_handles_labels()
        custom_line = Line2D([0], [0], color='yellow', lw=2, label='Fitted ellipse')
        handles.append(custom_line)
        labels.append('Fitted ellipse')
        plt.imshow(ellipse_image, cmap=ListedColormap(['black', 'yellow']), alpha=0.5)  # Plot the fitted ellipse
    
        # Title and labels
        plt.title(str(self))
        plt.xlim(x_hull_lim)
        plt.ylim(y_hull_lim)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(handles=handles, labels=labels, fontsize='xx-small', framealpha=0.75)
        if section_img is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(section_img, cmap='gray')
            plt.contour(img_todense, levels=[0.5], colors='red', alpha=0.35)
            plt.title("Contour")
            plt.xlim(x_hull_lim)
            plt.ylim(y_hull_lim)
            plt.xlabel("x")
            plt.ylabel("y")
        
        plt.show()

        
    def save(self, path: str):
        """
        Save the cell to a file using pickle.
        
        Parameters:
        - path: Path to the file where the object will be saved.
        """
        with open(path, 'wb') as file:
            pickle.dump(self, file)

            
    def fit_ellipse(self) -> dict:
        """
        Fit an ellipse to the convex hull points of the binary image.
        Uses OpenCV's implementation of the Direct least square fitting of ellipses (Fitzgibbon)
        
        Returns:
        - ellipse_params: A dictionary containing the parameters of the fitted ellipse.
        """
        # Extract the coordinates of the boundary points        
        boundary_points = self.get_boundary_points()
        
        # Convert the points to the format required by cv2.fitEllipse
        points = np.column_stack((boundary_points[:, 1], boundary_points[:, 0])).astype(np.float32)
        
        # Fit the ellipse to the points
        ellipse = cv2.fitEllipse(points)
        
        # Extract the ellipse parameters
        (center_x, center_y), (major_axis_length, minor_axis_length), angle = ellipse
                
        ellipse_params = {
            'centroid': (center_y, center_x),
            'major_axis_length': major_axis_length,
            'minor_axis_length': minor_axis_length,
            'orientation': np.deg2rad(angle)
        }
        return ellipse_params


    def get_ellipse_image(self, thickness: int=1) -> np.ndarray:
        """
        Draw an ellipse on an image.
        
        Parameters:
        - thickness - Thickness of the ellipse boundary. -1 fills in the ellipse.
        
        Returns:
        - image_with_ellipse: the image with the ellipse drawn on it.
        """
        # Convert the image to uint8 if it is not already
        image = np.zeros(self.image.shape, dtype=np.uint8)
        
        # Convert the image to RGB if it is grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        ellipse_params = self.fit_ellipse()
        
        # Extract the ellipse parameters
        centroid = ellipse_params['centroid']
        major_axis_length = ellipse_params['major_axis_length']
        minor_axis_length = ellipse_params['minor_axis_length']
        orientation = np.rad2deg(ellipse_params['orientation'])
        
        # Draw the ellipse on the image
        image_with_ellipse = image.copy()
        cv2.ellipse(image_with_ellipse, (int(centroid[1]), int(centroid[0])), 
                    (int(major_axis_length / 2), int(minor_axis_length / 2)), 
                    orientation, 0, 360, (255, 255, 0), thickness)
        
        return image_with_ellipse
    

    def get_eccentricity(self) -> float:
        """
        Get the eccentricity of the fitted ellipse around the cell
        
        Returns:
        - eccentricity: Eccentricity of the fitted ellipse around the cell
        """
        ellipse_params = self.fit_ellipse()
        # Extract the semi-major and semi-minor axis lengths
        major_axis = ellipse_params['major_axis_length'] / 2
        minor_axis = ellipse_params['minor_axis_length'] / 2
        b = min(major_axis, minor_axis)
        a = max(major_axis, minor_axis)
        return np.sqrt(1 - (b / a) ** 2)
    
    
    def get_ellipse_dice(self) -> float:
        """
        Compute the Dice coefficient between the cell and the fitted ellipse.
        
        Returns:
        - dice_coefficient: Dice coefficient between the cell and the fitted ellipse.
        """
        ellipse_image = self.get_ellipse_image(thickness=-1)
        ellipse_image = cv2.cvtColor(ellipse_image, cv2.COLOR_BGR2GRAY)
        ellipse_image = ellipse_image.astype(bool)
        return compute_dice_coefficient(self.image, ellipse_image)
            
        
    def __str__(self) -> str:
        """
        Return a string representation of the cell.
        
        Returns:
        - str: String representation of the cell.
        """
        return f"Cell {self.z}-{self.index}"
    
    
    def __repr__(self) -> str:
        """
        Return a string representation of the cell.
        
        Returns:
        - str: String representation of the cell.
        """
        return self.__str__()