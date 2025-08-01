import ants
import nibabel as nib
import numpy as np
import os

def register_images(moving_image_path, fixed_image_path, output_path):
    # Load the moving and fixed images
    moving_image = ants.image_read(moving_image_path)
    fixed_image = ants.image_read(fixed_image_path)

    # Perform the registration
    print("Registering moving image to fixed image...")
    registration_result = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN')

    # Save the registered image
    ants.image_write(registration_result['warpedmovout'], output_path)
    print(f"Registered image saved to {output_path}")
    
    # Save transformation files
    fwdtransforms = registration_result['fwdtransforms']
    print(f"Forward transforms: {fwdtransforms}")

    return fwdtransforms, registration_result

def apply_transform_to_points(points, fwdtransforms, fixed_image):
    # points: numpy array of shape (N, 3), in physical space
    # ants.apply_transforms_to_points expects a pandas DataFrame
    import pandas as pd
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    transformed = ants.apply_transforms_to_points(
        dim=3,
        points=df,
        transformlist=fwdtransforms,
        whichtoinvert=[False]*len(fwdtransforms),
        reference=fixed_image
    )
    return transformed[['x', 'y', 'z']].values

if __name__ == "__main__":
    moving_image_path = os.path.join('data', 'moving_image.nii.gz')
    fixed_image_path = os.path.join('data', 'fixed_image.nii.gz')
    output_path = os.path.join('data', 'registered_image.nii.gz')
    points_path = os.path.join('data', 'points.npy')  # shape (N, 3), physical space

    fwdtransforms, fixed_image = register_images(moving_image_path, fixed_image_path, output_path)

    # Load points
    points = np.load(points_path)  # shape (N, 3)

    # Apply transform
    transformed_points = apply_transform_to_points(points, fwdtransforms, fixed_image)
    np.save(os.path.join('data', 'transformed_points.npy'), transformed_points)
    print("Transformed points saved.")