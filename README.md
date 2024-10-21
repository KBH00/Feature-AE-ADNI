# Feature Autoencoder in ADNI

1. **Convert DICOM Files**:
   - Run `convert.sh` to convert DICOM files to NIfTI format and apply skull stripping.
   - You can revise the INPUT and OUTPUT PATH in the convert.sh

2. **Set Up the Environment and Train the Model**:
   - Create the conda environment and activate it:
     ```bash
     conda env create -f environment.yml
     conda activate anomaly_detection
     ```
   - Execute `main.py` to train the feature autoencoder on the processed data.


3. **Visualization**:
   - To visualize the results, run `Inference.py`.
