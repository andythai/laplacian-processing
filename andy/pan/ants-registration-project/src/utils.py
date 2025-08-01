import numpy as np
import nibabel as nib
import ants

def load_nifti_image(filepath):
    """Load a NIfTI image from a given file path."""
    return nib.load(filepath).get_fdata()

def save_nifti_image(data, filepath, affine=None):
    """Save a NIfTI image to a given file path."""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filepath)

def load_points(filepath):
    """Load corresponding points from a .npy file."""
    return np.load(filepath)

def save_points(points, filepath):
    """Save transformed points to a .npy file."""
    np.save(filepath, points)

def apply_transform_to_points(points, transform):
    """Apply the transformation to the array of points."""
    transformed_points = ants.apply_transforms(fixed=transform['fixed'], moving=points, type_of_transform='Affine')
    return transformed_points