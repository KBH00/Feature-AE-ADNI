import os
import nibabel as nib
import pydicom
import numpy as np
from torchvision import transforms
from skimage.transform import resize

def find_dcm_directories(base_dir):
    """
    Recursively find all directories containing .dcm files.

    Args:
        base_dir (str): The base directory to search.

    Returns:
        list: List of directories containing at least one .dcm file.
    """
    dcm_directories = []
    for root, dirs, files in os.walk(base_dir):
        if any(file.endswith(".dcm") for file in files):
            dcm_directories.append(root)
    return dcm_directories

def dicom_to_nifti(dcm_directory):
    """
    Convert all DICOM files in the given directory to a NIfTI file.

    Args:
        dcm_directory (str): Directory containing the DICOM files.

    Returns:
        str: Path to the generated NIfTI file.
    """
    dcm_files = sorted([f for f in os.listdir(dcm_directory) if f.endswith(".dcm")])
    print(dcm_files)
    if not dcm_files:
        raise ValueError(f"No DICOM files found in directory: {dcm_directory}")

    # Read all DICOM slices and stack them to create a 3D volume
    slices = []
    for dcm_file in dcm_files:
        file_path = os.path.join(dcm_directory, dcm_file)
        dicom_data = pydicom.dcmread(file_path)
        slice_image = dicom_data.pixel_array
        if slice_image.ndim == 2:  
            slice_image_resized = resize(slice_image, (128, 128), anti_aliasing=True)
        elif slice_image.ndim == 3: 
            slice_image_resized = resize(slice_image, (128, 128, 128), anti_aliasing=True)
        else:
            raise ValueError(f"Unexpected slice dimensions: {slice_image.ndim} for file: {dcm_file}")
        slices.append(slice_image_resized)

    nii_name_dir = dcm_directory.replace('/', '_')
    nii_name_dir = nii_name_dir.replace('\\', '_')

    if slices[0].ndim == 2:
        volume = np.stack(slices, axis=-1)  # Shape: (num_slices, 128, 128)
    else:
        volume = slices[0]  

    #volume = np.stack(slices, axis=-1) 
    nifti_image = nib.Nifti1Image(volume, np.eye(4))
    nifti_file_path = os.path.join(dcm_directory, "converted.nii")
    nib.save(nifti_image, nifti_file_path)

    print(f"NIfTI file saved at: {nifti_file_path}")
    return nifti_file_path

#D:/VascularData/data/ADNI
#D:/Data/FLAIR_T2_ss/ADNI

if __name__ == "__main__":
    path = "/home/kbh/Downloads/ADNI"
    train_directories = find_dcm_directories(path)
    
    for dic in train_directories:
        dicom_to_nifti(dic)