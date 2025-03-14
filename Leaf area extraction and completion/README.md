# Leaf area extraction

## Requirements
### MATLAB (R2018 or later recommended)
### Point Cloud Data in the file point.txt with at least 3 columns representing the X, Y, and Z coordinates.

## Key Parameters
 li: Adjustment coefficient for scaling the point cloud. The value should be between 0 and 1.
 
 samplingRate: Sampling ratio for downsampling the point cloud. The value should be between 0 and 1.

## Inputs
 Point Cloud Data: The input data is read from a file called point.txt, which should contain the 3D coordinates of the point cloud (X, Y, Z) in a text format.

## Outputs
 3D Mesh Plots: 3D visualizations are created using trimesh to show the surface mesh before and after processing.
## Usage Instructions
 Load Point Cloud Data: Ensure that your point cloud data file (point.txt) is available in the same directory as the script or specify the correct path in the importdata function.

## Adjust Parameters:
 Set the adjustment coefficient (li) to scale the point cloud as required.
 
 Adjust the sampling rate (samplingRate) to downsample the point cloud.
 
Run the Script: Execute the script in MATLAB.