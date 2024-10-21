# Feature Autoencoder in ADNI

1. **Convert DICOM Files**:
   - Run `convert.sh` to convert DICOM files to NIfTI format and apply skull stripping.
   - You can revise the INPUT and OUTPUT PATH in the convert.sh

2. **Train the Model**:
   - Execute `main.py` to train the feature autoencoder on the processed data.

3. **Visualization**:
   - To visualize the results, run `Inference.py`.
