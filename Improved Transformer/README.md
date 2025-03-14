#Transformer based segementation network
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