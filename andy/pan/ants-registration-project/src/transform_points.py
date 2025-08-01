import numpy as np
import ants

def transform_points(points, transformation_matrix):
    """
    Apply the transformation matrix to the array of points.

    Parameters:
    - points: np.ndarray, an array of shape (N, 3) where N is the number of points.
    - transformation_matrix: np.ndarray, a 4x4 transformation matrix obtained from ANTs registration.

    Returns:
    - transformed_points: np.ndarray, the transformed points in the fixed image space.
    """
    # Convert points to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack((points, ones))

    # Apply the transformation
    transformed_points = homogeneous_points @ transformation_matrix.T

    # Return only the x, y, z coordinates
    return transformed_points[:, :3]