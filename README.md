# Transformer based segementation network
## Part Segmentation
### Data Preparation
Follow these steps to prepare your data and initialize model training:

1. **Directory Structure Setup**  
   Organize your data according to the standard structure:  
   ```bash
   /data/maize/
   ├── point_cloud_files/  
   ├── synsetoffset2category.txt  
   └── train_test_split/
2.Essential Configuration Updates
- **Map semantic classes in**:  
  `synsetoffset2category.txt`
- **Define dataset splits in**:  
  `train_test_split`
### train
Change which method to use in `config/partseg.yaml` and run
```
python train_partseg.py
```
### test
We present a fully trained segmentation network and provide a corn point cloud dataset sample for evaluation purposes.
```
python test.py
```

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
