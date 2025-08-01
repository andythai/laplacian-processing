""" 
Code to visualize various image formats in Neuroglancer locally.
Maintained by Andy Thai (andy.thai@uci.edu)
"""
# Standard library
import webbrowser
import glob 
import sys
import copy
import inspect

# 3rd party libraries
import numpy as np
from natsort import natsorted
import neuroglancer
import zarr
from joblib import Parallel, delayed
import nibabel as nib

# Our libraries
from tif_to_ome import readTifWrapper, readTifSection


def image_shader():
    shader_code = """
    #uicontrol invlerp normalized(range=[0, 945], window=[0, 65535])
    void main() {  
        emitGrayscale(normalized());
    }
    """
    return inspect.cleandoc(shader_code)


def volume_shader(registered: bool=False):
    if not registered:  # If not using registered data, use the following shader code
        shader_code = """
        // Controls the strength of the volume rendering opacity
        #uicontrol float brightness slider(min=-10, max=10, default=0, step=0.1)

        // Settings for brain tissue
        #uicontrol float tissueMinValue slider(min=0, max=65535, default=0, step=1)
        #uicontrol float tissueMaxValue slider(min=0, max=65535, default=400, step=1)
        #uicontrol vec3 tissueColor color(default=\"#FFFFFF\")

        // Settings for cell values
        #uicontrol float cellMinValue slider(min=0, max=65535, default=500, step=1)
        #uicontrol vec3 cellColor color(default=\"#FFFF00\")

        void main() {
            // Get data and normalize it
            float x = float(getInterpolatedDataValue().value);
            float norm = (x - tissueMinValue) / (tissueMaxValue - tissueMinValue);
            
            // Default color
            vec4 voxelColor = vec4(tissueColor, norm*exp(brightness));
            
            // Color yellow if cell activation detected
            if (x > cellMinValue) {
                voxelColor = vec4(cellColor, norm*exp(brightness));
            }
            emitRGBA(voxelColor);
        }
        """
    else:  # If using registered data, use the following shader code. Only changes are in float x line.
        shader_code = """
        // Controls the strength of the volume rendering opacity
        #uicontrol float brightness slider(min=-10, max=10, default=0, step=0.1)

        // Settings for brain tissue
        #uicontrol float tissueMinValue slider(min=0, max=65535, default=0, step=1)
        #uicontrol float tissueMaxValue slider(min=0, max=65535, default=400, step=1)
        #uicontrol vec3 tissueColor color(default=\"#FFFFFF\")

        // Settings for cell values
        #uicontrol float cellMinValue slider(min=0, max=65535, default=500, step=1)
        #uicontrol vec3 cellColor color(default=\"#FFFF00\")

        void main() {
            // Get data and normalize it
            float x = float(getInterpolatedDataValue());
            float norm = (x - tissueMinValue) / (tissueMaxValue - tissueMinValue);
            
            // Default color
            vec4 voxelColor = vec4(tissueColor, norm*exp(brightness));
            
            // Color yellow if cell activation detected
            if (x > cellMinValue) {
                voxelColor = vec4(cellColor, norm*exp(brightness));
            }
            emitRGBA(voxelColor);
        }
        """
    return inspect.cleandoc(shader_code)


def rgba_shader():
    shader_code = """
    #uicontrol float brightness slider(min=0, max=100, default=1, step=0.1)
    #uicontrol float red_channel slider(min=0, max=1, default=1, step=0.01)
    #uicontrol float green_channel slider(min=0, max=1, default=1, step=0.01)
    #uicontrol float blue_channel slider(min=0, max=1, default=1, step=0.01)

    void main() {
        float r = toNormalized(getDataValue(0)) * red_channel;
        float g = toNormalized(getDataValue(1)) * blue_channel;
        float b = toNormalized(getDataValue(2)) * green_channel;
        vec3 color = vec3(r, g, b);
        color *= brightness;
        vec4 rgba = vec4(color, 1.0);
        emitRGBA(rgba);
    }
    """
    return inspect.cleandoc(shader_code)


def segmentation_shader():
    shader_code = """
    #uicontrol float brightness slider(min=-1, max=1, default=0)
    #uicontrol float contrast slider(min=0, max=2, default=1)
    
    void main() {
    // Get the integer value from the segmentation volume
    uint value = uint(getDataValue());

    // Generate a random color based on the value
    float r = fract(sin(float(value) * 12.9898) * 43758.5453);
    float g = fract(sin(float(value) * 78.233) * 43758.5453);
    float b = fract(sin(float(value) * 39.346) * 43758.5453);

    // Apply brightness and contrast adjustments
    vec3 color = vec3(r, g, b);
    color = (color - 0.5) * contrast + 0.5 + brightness;

    emitRGB(color);
    }
    """
    return inspect.cleandoc(shader_code)


def create_source(url, 
                  xyz_dim_in: list, dim_unit_in: list, 
                  xyz_dim_out: list, dim_unit_out: list, 
                  transform_matrix: list):
    """Generates a source dictionary for Neuroglancer.

    Args:
        url (_type_): URL or image layer 
        xyz_dim (list): Unit dimensions of the image
        dim_unit (str): Unit of the dimensions
        xyz_mat (list): Diagonal scaling elements for the affine matrix for the image

    Returns:
        dict: source dictionary for Neuroglancer
    """
    x_dim_in, y_dim_in, z_dim_in = xyz_dim_in
    x_unit_in, y_unit_in, z_unit_in = dim_unit_in
    x_dim_out, y_dim_out, z_dim_out = xyz_dim_out
    x_unit_out, y_unit_out, z_unit_out = dim_unit_out
    #x_mat, y_mat, z_mat = xyz_mat
    #transform_matrix = [[x_mat, 0, 0, 0],
    #                    [0, y_mat, 0, 0],
    #                    [0, 0, z_mat, 0]]
    # Generate source dictionary
    src = {
        "url": url,
        "transform": {
            "outputDimensions": {
                "x": [x_dim_out, x_unit_out],
                "y": [y_dim_out, y_unit_out],
                "z": [z_dim_out, z_unit_out]
            },
            "inputDimensions": {
                "x": [x_dim_in, x_unit_in],
                "y": [y_dim_in, y_unit_in],
                "z": [z_dim_in, z_unit_in]
            },
            "matrix": transform_matrix,
        },
    }
    return src


def create_image_layer(input_path, dimensions):
    url = input_path  # Used to generate the layers later. Will be an actual URL if a URL is given. 
                      # Otherwise, it will be a LocalVolume object.

    # Check if input path is a URL
    if '://' in input_path:
        print(f"Loading OME-Zarr image from URL: {input_path}")
        # Create image layer from URL
        image_layer = neuroglancer.ImageLayer(
            source=create_source(input_path, [x_dim, y_dim, z_dim], dim_unit, [x_mat, y_mat, z_mat]),
            opacity=1.0,
            shader=image_shader(),
        )
        
    # Otherwise, load up the image locally
    else:
        # Case: input path is a nii file, so load up nii file
        if input_path.endswith('.nii') or input_path.endswith('.nii.gz'):
            print(f"Loading nii file from {input_path}")
            input_volume = nib.load(input_path).get_fdata()
            print(f"\tImage shape: {input_volume.shape}")
            z, h, w = input_volume.shape
            
        # Case: input path is a directory, so load up tiff files locally
        else:
            print(f"Loading tiff files from {input_path}")
            fns = natsorted(glob.glob(input_path + "/*.tif", recursive=True))
            image = readTifSection(str(fns[0]))
            w, h = image.shape
            z = len(fns)
            print(f"\tImage shape: {z} x {w} x {h}")
            input_volume = zarr.open(input_path, shape=(z, w, h), chunks=(1, w, h), dtype='u2')
            Parallel(n_jobs=5, verbose=13)(delayed(readTifWrapper)(i, input_volume, fn) for i, fn in enumerate(fns))
            input_volume = input_volume[:]  # Converts to NumPy array
            
        
        image_layer = neuroglancer.LocalVolume(data=input_volume, 
                                               dimensions=dimensions, 
                                               volume_type='image', 
                                               voxel_offset=[0, 0, 0])
        url = image_layer
    return image_layer, url, (z, h, w)


def create_annotation_template_layer(segmentation_path, dimensions):
    print(f"Loading segmentation annotation from {segmentation_path}")
    segmentation_volume = nib.load(segmentation_path).get_fdata()
    print(f"\tAtlas segmentation shape: {segmentation_volume.shape}")
    segmentation_layer = neuroglancer.LocalVolume(
        data=segmentation_volume,
        dimensions=dimensions,
        volume_type='image',
        voxel_offset=[0, 0, 0],
    )
    return segmentation_layer


def create_points_layer(points_path, annotation_color, registered=False):
    print(f"Loading input points from {points_path}")
    points = np.loadtxt(points_path, skiprows=2)  # Adjust skiprows if necessary
    # Invert the z-axis of the points to match the Neuroglancer coordinate system
    if registered:
        annotations = [neuroglancer.PointAnnotation(id=f'point{i}', point=[point[2], point[1], point[0]]) 
                    for i, point in enumerate(points)]
    else:
        annotations = [neuroglancer.PointAnnotation(id=f'point{i}', point=[point[2], point[1], z - point[0] - 1]) 
                    for i, point in enumerate(points)]
    annotation_layer = neuroglancer.AnnotationLayer(annotations=annotations,
                                                    annotation_color=annotation_color)
    return annotation_layer


if __name__ == '__main__':
    # Input localhost address and port
    ADDRESS = "0.0.0.0"
    PORT = 8080
    
    REGISTERED = True # If true, tries to load the registered data
    
    # Input paths or parameters
    #input_path = f"zarr://http://{ADDRESS}:{PORT}/output/B0039_ome.zarr_0/ome.zarr/"  # Locally hosted URL
    #input_path = f"zarr://http://{ADDRESS}:{PORT}/output/B0039_rgb_ome.zarr/ome.zarr/"  # Locally hosted URL (RGB)
    input_path = "zarr://https://kiwi1.ics.uci.edu/neuroglancer-ftp/B0039/ome.zarr/"  # FTP URL
    #input_path = "../B0039_stitched/stitched_ch0/"  # TIF files
    
    # Input point annotations and misc. data to visualize
    points_path = "output/inputpoints.txt"  # Cleared points in original resolution for B0039
    segmentation_path = None
    
    # Registered data
    if REGISTERED:
        input_path = "output/elastixRefined.nii.gz"  # nii.gz
        points_path = "output/outputIndices.txt"
        segmentation_path = "CCF_DATA/annotation_25.nii.gz"
        
        # Lightsheet
        input_path = "../Eric_bloodvessel_lightsheet_data/output/result_brightened.nii"
        points_path = ""
        segmentation_path = "../Eric_bloodvessel_lightsheet_data/processed/smartspim_annotation_cropped.nii"

    # Scaling parameters
    xyz_dim = [0.00125, 0.00125, 0.05]  # 1250, 1250, 50000 micrometers
    xyz_unit = ["m", "m", "m"]
    x_dim, y_dim, z_dim = xyz_dim  # 1250, 1250, 50000 micrometers
    x_unit, y_unit, z_unit = xyz_unit
    
    # For the affine matrix
    if REGISTERED:
        x_mat, y_mat, z_mat = [24.6403508772, 27.2375, 1]  # For registered data
    else:
        x_mat, y_mat, z_mat = [1, 1, 1]
    
    # Default width and height since no information is available from just URLs
    z, h, w = 280, 8716, 11236
    
    # Names and information for the layers
    IMAGE_LAYER_NAME = 'image layer'
    VOLUME_LAYER_NAME = 'volume visualization'
    ANNOTATIONS_LAYER_NAME = 'point annotations'
    ANNOTATION_COLOR = '#ff0000'  # Red color for annotations
    SEGMENTATION_LAYER_NAME = 'allen mouse ccf'
    
    ################################################################
    # Setup the data sources
    ################################################################
    
    # Initialize Neuroglancer
    neuroglancer.set_server_bind_address(ADDRESS, PORT)
    
    dimensions = neuroglancer.CoordinateSpace(names=['z', 'y', 'x'], 
                                              units=[z_unit, y_unit, x_unit], 
                                              scales=[z_dim, y_dim, x_dim])
    
    image_layer, url, image_dim = create_image_layer(input_path, dimensions)
    z, h, w = image_dim
    
    # Make a volume visualization layer, which is a copy of the original layer
    volume_layer = copy.deepcopy(image_layer)
    
    # Load input points
    if points_path:
        annotation_layer = create_points_layer(points_path, ANNOTATION_COLOR, REGISTERED)
    
    # Load Allen CCF 2017 segmentation annotation
    if segmentation_path:
        segmentation_layer = create_annotation_template_layer(segmentation_path, dimensions)
        
    ################################################################
    #  Setup the Neuroglancer viewer
    ################################################################
    
    viewer = neuroglancer.Viewer()
    
    # Add layers to the viewer
    with viewer.txn() as s:
        # Add image layer
        s.layers.append(
            name=IMAGE_LAYER_NAME,
            source=create_source(url, 
                                 [x_dim, y_dim, z_dim], xyz_unit, 
                                 [x_dim, y_dim, z_dim], xyz_unit, 
                                 [x_mat, y_mat, z_mat]),
            layer=image_layer,
            opacity=1.0,
            shader=image_shader(),
        )

        # Add volume visualization layer
        s.layers.append(
            name=VOLUME_LAYER_NAME,
            source=create_source(url, 
                                 [x_dim, y_dim, z_dim], xyz_unit, 
                                 [x_dim, y_dim, z_dim], xyz_unit, 
                                 [x_mat, y_mat, z_mat]),
            layer=volume_layer,
            opacity=0,
            shader=volume_shader(registered=REGISTERED),
            shaderControls={
                "brightness": -1.5,
                "tissueMinValue": 40,
            },
            blend="additive",
            volume_rendering=True,
            volumeRenderingGain=-4.6 if REGISTERED else -7.1,
        )
        
        # Add annotations layer if points are provided
        if points_path:
            s.layers.append(
                name=ANNOTATIONS_LAYER_NAME,
                source=create_source("local://annotations", 
                                     [x_dim, y_dim, z_dim], xyz_unit,
                                     [x_dim, y_dim, z_dim], xyz_unit,
                                     [x_mat, y_mat, z_mat]),
                layer=annotation_layer,
            )
            
        # Add segmentation layer
        if segmentation_path:
            s.layers.append(
                name=SEGMENTATION_LAYER_NAME,
                source=create_source(segmentation_layer, 
                                     [x_dim, y_dim, z_dim], xyz_unit,
                                     [x_dim, y_dim, z_dim], xyz_unit,
                                     [x_mat, y_mat, z_mat]),
                layer=segmentation_layer,
                shader=segmentation_shader(),
            )
                
        # Setup viewer settings
        s.voxel_coordinates = [int(w / 2 - 1) * x_mat, int(h / 2 - 1) * y_mat, int(z / 2 - 1) * z_mat]  # Starting location of camera
        s.dimensions = neuroglancer.CoordinateSpace(names=["x", "y", "z"],
                                                    units=xyz_unit,
                                                    scales=[x_dim, y_dim, z_dim])
        s.projection_scale = 13107.2
        s.cross_section_scale = 16.068429538550138  # How zoomed in the preview is
        s.projection_orientation = [-0.2444217950105667, 
                                    -0.01222456619143486, 
                                    -0.011153397150337696, 
                                     0.9695277810096741]
        s.selectedLayer.layer = IMAGE_LAYER_NAME
        s.selectedLayer.visible = True
        s.selectedLayer.size = 1162
    
    print("\nGenerated local Neuroglancer URL:", viewer)
    webbrowser.open(viewer.get_viewer_url())
    input("Press ENTER to end the visualization...")
    sys.exit(0)