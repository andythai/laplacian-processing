import matplotlib.pyplot as plt
import numpy as np
from modules.laplacian_fixed import createA as createA_fixed
from modules.laplacian import createA
from scipy.ndimage import geometric_transform
from functools import partial


def compute_error(resolution, mpoints, fpoints, deformation):
    """Compute the error between the deformation and the expected values."""
    A, Zd, Yd, Xd = createA(np.zeros((1, resolution[0], resolution[1])), mpoints, fpoints)
    dx = deformation[2, 0, :, :]
    dy = deformation[1, 0, :, :]
    dz = deformation[0, 0, :, :]

    residual_x = A @ dx.flatten() - Xd
    residual_y = A @ dy.flatten() - Yd
    residual_z = A @ dz.flatten() - Zd

    err_x = np.linalg.norm(residual_x)
    err_y = np.linalg.norm(residual_y)
    err_z = np.linalg.norm(residual_z)
    total_err = np.linalg.norm(residual_x + residual_y + residual_z)
    return err_x, err_y, err_z, total_err


def compute_error_fixed(resolution, mpoints, fpoints, deformation):
    """Compute the error between the deformation and the expected values."""
    A, Zd, Yd, Xd = createA_fixed(np.zeros((1, resolution[0], resolution[1])), mpoints, fpoints)
    dx = deformation[2, 0, :, :]
    dy = deformation[1, 0, :, :]
    dz = deformation[0, 0, :, :]

    residual_x = A @ dx.flatten() - Xd
    residual_y = A @ dy.flatten() - Yd
    residual_z = A @ dz.flatten() - Zd

    err_x = np.linalg.norm(residual_x)
    err_y = np.linalg.norm(residual_y)
    err_z = np.linalg.norm(residual_z)
    total_err = np.linalg.norm(residual_x + residual_y + residual_z)
    return err_x, err_y, err_z, total_err


def correspondence_error(deformation, moving_points, fixed_points):
    """
    Compute the Euclidean error at each correspondence after applying the deformation.
    Args:
        deformation: np.ndarray, shape (channels, batch, H, W)
        moving_points: np.ndarray, shape (N, 3) -- (z, y, x) or (y, x)
        fixed_points: np.ndarray, shape (N, 3) -- (z, y, x) or (y, x)
    Returns:
        errors: np.ndarray, shape (N,) -- Euclidean distance for each correspondence
    """
    # Assume batch=0, channels=[z,y,x] or [y,x]
    # If 2D, use channels 1 and 2 (y, x)
    transformed_points = []
    for i in range(moving_points.shape[0]):
        y, x = moving_points[i, 1], moving_points[i, 2]
        dy = deformation[1, 0, int(y), int(x)]
        dx = deformation[2, 0, int(y), int(x)]
        new_y = y + dy
        new_x = x + dx
        transformed_points.append([moving_points[i, 0], new_y, new_x])
    transformed_points = np.array(transformed_points)
    errors = np.linalg.norm(transformed_points - fixed_points, axis=1)
    return errors


def shift3Dfunc(point, dx, dy, dz):
    try:
        px = point[0] + dx[point[0], point[1],  point[2]]
        py = point[1] + dy[point[0], point[1],  point[2]]
        pz = point[2] + dz[point[0], point[1],  point[2]]
        if(px<0 or px> dx.shape[0]):
            return (point[0], point[1], point[2])
        if(py<0 or py> dx.shape[1]):
            return (point[0], point[1], point[2])
        if(pz<0 or pz> dx.shape[2]):
            return (point[0], point[1], point[2])
        return (px, py, pz)
    except:
        return (0, 0, 0)

def apply_transformation_fixed(d, moving_image, mpoints=None, fpoints=None, title="Transformed image"):
    #print(moving_image.shape)
    ch0 = moving_image[:, :]
    ch0 = ch0[np.newaxis, :, :]  # Add a new axis for the channel dimension

    deformation_field = d.deformation  # Get the deformation field (dx, dy, dz)
    
    # Convert all displacements to rounded integers
    #deformation_field = np.round(deformation_field).astype(np.int32)
    
    transformed_ch0 = geometric_transform(ch0, 
                                          partial(shift3Dfunc, dx=deformation_field[0], dy=deformation_field[1], dz=deformation_field[2]), order=1, prefilter=False)
    transformedData = transformed_ch0  # Use only the first channel for now
    transformedData = transformedData[0, :, :]  # Get the first channel only
    # Binarize the transformed image
    #transformedData = (transformedData > 0.5).astype(np.float32)  # Binarize the image
    
    plt.figure(figsize=(10, 10))
    plt.imshow(transformedData, cmap='gray')
    
    # Overlay the moving image points and fixed image points
    if mpoints is not None and fpoints is not None:
        for i in range(mpoints.shape[0]):
            #plt.arrow(mpoints[i, 2], mpoints[i, 1], fpoints[i, 2] - mpoints[i, 2], fpoints[i, 1] - mpoints[i, 1],
            #          head_width=0.15, head_length=0.15, fc='b', ec='b', alpha=0.5)
            #plt.text(mpoints[i, 2], mpoints[i, 1] + 0.4, str(i), color='green', ha='center', va='bottom', fontsize=8, weight='bold', alpha=0.25)
            plt.scatter(mpoints[i, 2], mpoints[i, 1], color='red', s=5, alpha=0.25)
            plt.scatter(fpoints[i, 2], fpoints[i, 1], color='green', s=5, alpha=0.1)
    plt.title(title)
    plt.legend(['Moving Points', 'Fixed Points'], loc='upper right')
    plt.show()
    return transformedData


def apply_transformation_forward_splat(d, moving_image, mpoints=None, fpoints=None, title="Transformed image"):
    """
    Forward mapping for binary images: splat black wherever any black pixel lands.
    """
    deformation_field = d.deformation  # (channels, batch, y, x)
    H, W = moving_image.shape
    out_img = np.ones_like(moving_image, dtype=np.float32)  # Start with all white

    for y in range(H):
        for x in range(W):
            val = moving_image[y, x]
            if val > 0.5:
                continue  # Only splat black pixels

            dy = deformation_field[1, 0, y, x]
            dx = deformation_field[2, 0, y, x]
            new_y = y + dy
            new_x = x + dx

            # Bilinear splat to output image
            if 0 <= new_y < H-1 and 0 <= new_x < W-1:
                y0 = int(np.floor(new_y))
                x0 = int(np.floor(new_x))
                
                y0 = int(np.round(new_y))
                x0 = int(np.round(new_x))
                
                y1 = y0 + 1
                x1 = x0 + 1
                wy = new_y - y0
                wx = new_x - x0
                # Splat black (0) using min (so any hit makes it black)
                if 0 <= y0 < H and 0 <= x0 < W:
                    out_img[y0, x0] = min(out_img[y0, x0], 0)
                if 0 <= y0 < H and 0 <= x1 < W:
                    out_img[y0, x1] = min(out_img[y0, x1], 0)
                if 0 <= y1 < H and 0 <= x0 < W:
                    out_img[y1, x0] = min(out_img[y1, x0], 0)
                if 0 <= y1 < H and 0 <= x1 < W:
                    out_img[y1, x1] = min(out_img[y1, x1], 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(out_img, cmap='gray', vmin=0, vmax=1)
    if mpoints is not None and fpoints is not None:
        for i in range(mpoints.shape[0]):
            plt.scatter(mpoints[i, 2], mpoints[i, 1], color='green', s=5, alpha=0.25)
            plt.scatter(fpoints[i, 2], fpoints[i, 1], color='red', s=5, alpha=0.1)
    plt.title(title)
    plt.legend(['Moving Points', 'Fixed Points'], loc='upper right')
    plt.show()
    return out_img


def apply_transformation(d, moving_image, mpoints=None, fpoints=None, title="Transformed image"):
    """
    Forward mapping for binary images: splat black wherever any black pixel lands (nearest neighbor).
    """
    deformation_field = d.deformation  # (channels, batch, y, x)
    H, W = moving_image.shape
    out_img = np.ones_like(moving_image, dtype=np.float32)  # Start with all white

    for y in range(H):
        for x in range(W):
            val = moving_image[y, x]
            if val > 0.5:
                continue  # Only splat black pixels

            dy = deformation_field[1, 0, y, x]
            dx = deformation_field[2, 0, y, x]
            new_y = int(np.round(y + dy))
            new_x = int(np.round(x + dx))

            if 0 <= new_y < H and 0 <= new_x < W:
                out_img[new_y, new_x] = 0  # Paint black

    plt.figure(figsize=(10, 10))
    plt.imshow(out_img, cmap='gray', vmin=0, vmax=1)
    if mpoints is not None and fpoints is not None:
        for i in range(mpoints.shape[0]):
            plt.scatter(mpoints[i, 2], mpoints[i, 1], color='green', s=5, alpha=0.25)
            plt.scatter(fpoints[i, 2], fpoints[i, 1], color='red', s=5, alpha=0.1)
        plt.legend(['Moving Points', 'Fixed Points'], loc='upper right')
    plt.title(title)
    plt.show()
    return out_img


def deformation_heatmap(d, title="Deformation Heatmap"):
    """
    Show side-by-side heatmaps of the Y and X deformation fields.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    im0 = axes[0].imshow(d.deformation[1, 0], cmap='coolwarm', vmin=-np.max(np.abs(d.deformation[1, 0])), vmax=np.max(np.abs(d.deformation[1, 0])))
    axes[0].set_title('Y Deformation (pixels)')
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(d.deformation[2, 0], cmap='coolwarm', vmin=-np.max(np.abs(d.deformation[2, 0])), vmax=np.max(np.abs(d.deformation[2, 0])))
    axes[1].set_title('X Deformation (pixels)')
    axes[1].set_xlabel('X coordinate')
    axes[1].set_ylabel('Y coordinate')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.show()
    
    