import pandas as pd
import subprocess
import numpy as np
import os
import SimpleITK as sitk
import time

from run_registration_cellcounting import registration, get_input_cell_locations, \
                                          parsePhysicalPointsFromOutputFile, convertPhysicalPointsToIndex, loadTransformMapping, getMappedIndices, \
                                          countCellsInRegions
from cellAnalysis.cell_counting import process_counts

# Run main
if __name__ == '__main__':
    # Define paths
    annotationImagePath = r"./CCF_DATA/annotation_25.nii.gz"
    ATLAS_PATH = r"./CCF_DATA/1_adult_mouse_brain_graph_mapping.csv"
    INPUT_FILEPATH = "gui test 2//"
    OUTPUT_FILEPATH = "mapping output//"
    
    # Load in outputted data from registration
    input_points = INPUT_FILEPATH + "//inputpoints_scaled.txt"
    scaledCellLocations = np.loadtxt(input_points, skiprows=2)


    start = time.time()
    #mapping = loadTransformMapping(fData.shape, mData.shape, REGISTERED_OUTPUT_PATH)
    #np.save("mapping.npy", mapping)
    cell_count_file = os.path.join(INPUT_FILEPATH, "cell_count.csv")
    annotationImage  = sitk.ReadImage(annotationImagePath)
    mapping = np.load(INPUT_FILEPATH + "//mapping.npy")
    outputIndices = getMappedIndices(scaledCellLocations, mapping)

    print("Cell location transformation done in {}".format(time.time()- start))

    #np.savetxt(output_points_file, scaledCellLocations , "%d %d %d", header = "index\n"+str(scaledCellLocations.shape[0]), comments ="")
    cellRegionCounts, pointIndices = countCellsInRegions(outputIndices, annotationImage)
    pd.DataFrame(dict(cellRegionCounts).items(), columns=["region", "count"]).to_csv(cell_count_file, index=False)
    atlas_df = pd.read_csv(ATLAS_PATH, index_col=None)
    count_df = pd.read_csv(cell_count_file, index_col=None)
    region_df,count_df = process_counts(atlas_df, count_df)
    count_df.to_csv(os.path.join(OUTPUT_FILEPATH, "cell_region_count.csv"), index=False)
    region_df.to_csv(os.path.join(OUTPUT_FILEPATH, "region_counts.csv"), index=False)
