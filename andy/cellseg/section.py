# Section class for processing and storing cell information
# Author: Andy Thai
# Standard library imports
from collections import defaultdict
import pickle
import sys
sys.path.append('../../')
import time

# 3rd party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.ndimage import median_filter
import SimpleITK as sitk
from skimage.measure import label, regionprops
import tqdm

# Our libraries
from cell import Cell
from gui.gui_shared import auto_contrast


# Extract the binary image for the biggest connected component
def get_component_binary_image(labeled_image: np.ndarray, component_index: int) -> np.ndarray:
    """
    Extract the binary image for a specific connected component index.
    
    Parameters:
    - labeled_image: Labeled image with connected components.
    - component_index: Index of the connected component to extract.
    
    Returns:
    - component_binary_image: Binary image of the extracted component.
    """
    component_binary_image = (labeled_image == component_index).astype(np.uint8)
    return component_binary_image


def n4_bias_correction(img, alpha: float=1, shrink_factor: float=15, show: bool=False) -> np.ndarray:
    """
    N4 bias correction for the input image.
    
    Parameters:
    - img: The input image to correct.
    - alpha: The alpha value for contrast adjustment.
    - shrink_factor: The shrink factor for downsampling the image for bias correction.
    - show: Whether to show the intermediate results.
    
    Returns:
    - corrected_image_full_resolution: The bias corrected image.
    """
    # Get contrast image for mask
    contrast_img = auto_contrast(img, alpha=alpha)
    
    # Create the brain tissue mask
    mask_img = sitk.GetImageFromArray(contrast_img)
    mask_img = sitk.RescaleIntensity(mask_img, 0, 255)
    mask_img = sitk.LiThreshold(mask_img, 0, 1)

    # Use the raw image and convert it to float32
    raw_img = sitk.GetImageFromArray(img.copy())
    raw_img = sitk.Cast(raw_img, sitk.sitkFloat32)

    # Downsample it for bias correction
    inputImage = raw_img
    if shrink_factor > 1:
        inputImage = sitk.Shrink( raw_img, [ shrink_factor ] * raw_img.GetDimension() ) #2
        maskImage = sitk.Shrink( mask_img, [ shrink_factor ] * inputImage.GetDimension() ) #3

    # Run bias correction
    start_time = time.time()
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias_corrector.Execute(inputImage, maskImage)
    
    # Apply bias correction to full resolution image
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img)
    corrected_image_full_resolution = raw_img / sitk.Exp(log_bias_field)
    end_time = time.time()
    corrected_image_full_resolution = sitk.GetArrayFromImage(corrected_image_full_resolution)
    
    # Show the process if True
    if show:
        print(f"Time taken for bias correction: {end_time - start_time:.2f} seconds")
        
        # Show the brain tissue mask
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(sitk.GetArrayFromImage(mask_img), cmap='gray')
        plt.title(f"Full resolution brain mask")
        plt.subplot(1, 2, 2)
        plt.imshow(sitk.GetArrayFromImage(maskImage), cmap='gray')
        plt.title(f"Downsampled brain mask (shrink factor={shrink_factor})")
        plt.show()
        
        # Show the log bias field
        plt.figure(figsize=(10, 5))
        plt.imshow(sitk.GetArrayFromImage(log_bias_field))
        plt.colorbar()
        plt.title(f"Log bias field")
        plt.show()

        # Show the corrected bias field image
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Original raw image")
        plt.subplot(1, 2, 2)
        plt.imshow(corrected_image_full_resolution, cmap='gray')
        plt.title(f"Corrected bias raw image")
        plt.show()

        # Increase the contrast of the corrected image and show side-by-side
        preview_alpha = 0.25
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        contrast_comparison = auto_contrast(img, alpha=preview_alpha)
        plt.imshow(contrast_comparison, cmap='gray')
        plt.title(f"Original contrast image (alpha={preview_alpha})")
        plt.subplot(1, 2, 2)
        corrected_bias_contrast = auto_contrast(corrected_image_full_resolution, alpha=preview_alpha)
        plt.imshow(corrected_bias_contrast, cmap='gray')
        plt.title(f"Corrected bias contrast image (alpha={preview_alpha})")
        plt.show()
        
    return corrected_image_full_resolution


def preprocess_image(img, alpha: float=1, shrink_factor: float=15, 
                     median_filter_size: int=5, gaussian_sigma: float=0.2, 
                     show: bool=False) -> np.ndarray:
    """
    Preprocess the image using N4 bias correction and filtering.
    
    Parameters:
    - img: The input image to preprocess.
    - alpha: The alpha value for contrast adjustment.
    - shrink_factor: The shrink factor for downsampling the image for bias correction.
    - median_filter_size: The size of the median filter to apply.
    - gaussian_sigma: The sigma value for the Gaussian filter to apply.
    - show: Whether to show the intermediate results.
    
    Returns:
    - corrected_img: The preprocessed image.
    """
    corrected_bias_img = n4_bias_correction(img, alpha=alpha, shrink_factor=shrink_factor, show=show)

    # Run median filter
    median_filtered_img = median_filter(corrected_bias_img.copy(), size=median_filter_size)

    # Run gaussian filter
    gaussian_filtered_img = scipy.ndimage.gaussian_filter(median_filtered_img.copy(), sigma=gaussian_sigma)

    if show:
        get_stats(img, title="Original image stats:")
        get_stats(corrected_bias_img, title="N4 bias corrected image stats:")
        get_stats(median_filtered_img, title="Median filtered image stats:")
        get_stats(gaussian_filtered_img, title="Gaussian filtered image stats:")
        
    corrected_img = gaussian_filtered_img        
    return corrected_img


def get_stats(img: np.ndarray, title: str="", show: bool=True) -> tuple:
    """
    Print the statistics of the image.
    
    Parameters:
    - img: The input image to compute statistics on.
    - title: The title of the statistics.
    - show: Whether to show the statistics.
    
    Returns:
    - img_min: The minimum value of the image.
    - img_max: The maximum value of the image.
    - img_median: The median value of the image.
    - img_mean: The mean value of the image.
    - img_std: The standard deviation of the image.
    """
    img_min = np.min(img)
    img_max = np.max(img)
    img_median = np.median(img)
    img_mean = np.mean(img)
    img_std = np.std(img)
    if show:
        if title:
            print(title)
            prefix = "\t"
        else:
            prefix = ""
        print(f"{prefix}Shape:", img.shape)
        print(f"{prefix}Min:", img_min)
        print(f"{prefix}Max:", img_max)
        print(f"{prefix}Median:", img_median)
        print(f"{prefix}Mean:", img_mean)
        print(f"{prefix}Std:", img_std)
        print()
    return img_min, img_max, img_median, img_mean, img_std
    

class Section:
    def __init__(self, img: np.ndarray, index: int):
        """
        Initialize the section object with the image and index.
        
        Parameters:
        - img: The image of the section.
        - index: The index of the section.
        """
        self.image = img
        self.index = index  # Section index
        
        self.labeled_image = None
        self.cells = defaultdict()
        self.df = None

    
    def process(self, alpha: float=1, shrink_factor: float=15, 
                median_filter_size: int=5, gaussian_sigma: float=0.2, threshold_scale: float=20):
        """
        Process the section image and populate the regions and cells.
        
        Parameters:
        - alpha: The alpha value for contrast adjustment.
        - shrink_factor: The shrink factor for downsampling the image for bias correction.
        - median_filter_size: The size of the median filter to apply.
        - gaussian_sigma: The sigma value for the Gaussian filter to apply.
        - threshold_scale: The scale factor for thresholding to multiply with the standard deviation.
        """
        # Perform bias correction and filtering
        corrected_image = preprocess_image(self.image.copy(), alpha=alpha, shrink_factor=shrink_factor, 
                                           median_filter_size=median_filter_size, gaussian_sigma=gaussian_sigma, 
                                           show=False)
        
        # Compute thresholding
        _, _, _, img_mean, img_std = get_stats(corrected_image, show=False)
        threshold_value = img_mean + threshold_scale * img_std
        thresholded_img = np.where(corrected_image > threshold_value, 1, 0).astype(bool)
        
        # Label connected components
        self.labeled_image = label(thresholded_img)
        regions = regionprops(self.labeled_image)
        print(f"Number of regions found: {len(regions)}")
        
        # Populate dataframe with cell information
        df_data = []
        for region in tqdm.tqdm(regions, desc="Processing regions"):
            component_label = region.label
            component = get_component_binary_image(self.labeled_image, component_label)
            cell = Cell(component, z=self.index, index=component_label)
            self.cells[int(region.label)] = cell
            cell_area, convexity, eccentricity, ellipse_dice, _, _, _ = cell.get_stats(show=False)  # Shape, centroid, medoid
            df_data.append({'Label': component_label, 'Area': cell_area, 'Convexity': convexity, 'Eccentricity': eccentricity, 'Ellipse Dice': ellipse_dice})
        self.df = pd.DataFrame(df_data)
    

    def sort(self, key: str='area', ascending: bool=False) -> pd.DataFrame:
        """
        Sort the cells in the section based on the given key.
        
        Parameters:
        - key: The key to sort the cells by. Can be 'area', 'convexity', 'eccentricity', or 'ellipse dice'.
        - ascending: Whether to sort in ascending order.
        
        Returns:
        - df: The sorted dataframe of the section.
        """
        df = self.df.rename(columns=str.lower)
        df = df.sort_values(by=key.lower(), ascending=ascending)
        return df
        
        
    def get_df(self) -> pd.DataFrame:
        """
        Return the dataframe of the section.
        """
        return self.df
    
    
    def remove(self, key):
        """
        Remove the cell with the given key.
        
        Parameters:
        - key: The key of the cell to remove. Can be an integer, a list of integers, or a string.
        """
        if type(key) == str:  # If str, convert to int
            key = int(key)
        if type(key) == int:  # If int, remove the key
            self.df = self.df[self.df['Label'] != key]
            del self.cells[key]
            self.labeled_image[self.labeled_image == key] = 0
        if type(key) == list:  # If list, iterate through and remove all keys
            for k in key:
                k = int(k)
                self.df = self.df[self.df['Label'] != k]
                del self.cells[k]
                self.labeled_image[self.labeled_image == k] = 0        
    
    
    def save(self, path: str):
        """
        Save the section object to a pickle file.
        
        Parameters:
        - path: The path to save the pickle file.
        """
        with open(path, 'wb') as file:
            pickle.dump(self, file)
    
    
    def keys(self) -> list:
        """
        Return the keys of the cells in the section.
        
        Returns:
        - keys: List of keys of the cells in the section.
        """
        return self.cells.keys()
    
    
    def values(self) -> list:
        """
        Return the values of the cells in the section.
        
        Returns:
        - values: List of values of the cells in the section
        """
        return self.cells.values()
    
    
    def __len__(self) -> int:
        """
        Return the number of cells in the section.
        """
        return len(self.cells)
    
    
    def __str__(self) -> str:
        """
        Return the string representation of the section.
        """
        return f"Section {self.index}"
    
    
    def __repr__(self) -> str:
        """
        Return the string representation of the section.
        """
        return self.__str__()
    
    
    def __getitem__(self, key) -> Cell:
        """
        Access the cell with the given key.
        """
        return self.cells[key]