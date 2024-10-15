import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from functools import partial
from Data.data_utils import load_files_to_ram, load_nii_nn


class Nifti3DDataset(Dataset):
    def __init__(self, directories, transform=None, labels=None, config=None):
        """
        Custom dataset for loading 3D NIfTI images using `load_nii_nn`.

        Args:
            directories (list): List of directories containing .nii files.
            transform (callable, optional): Optional transform to be applied on a sample.
            labels (list, optional): Optional labels for supervised learning.
            config (Namespace, optional): Configuration containing loading parameters.
        """
        self.directories = directories
        self.transform = transform
        self.labels = labels
        self.config = config

        load_fn = partial(load_nii_nn,
                          slice_range=config.slice_range,
                          size=config.image_size,
                          normalize=config.normalize,
                          equalize_histogram=config.equalize_histogram)

        self.volumes = load_files_to_ram(self.directories, load_fn)

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        volume_resized = self.volumes[idx]

        if self.transform:
            volume_resized = self.transform(volume_resized)

        label = self.labels[idx] if self.labels is not None else -1
        return volume_resized, label

class TrainDataset(Dataset):
    """
    Training dataset. No anomalies, no segmentation maps.
    """

    def __init__(self, imgs: np.ndarray):
        """
        Args:
            imgs (np.ndarray): Training slices
        """
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]

class AugmentedTrainDataset(Dataset):
    """
    Training dataset with augmented images (horizontal and vertical flip).
    """

    def __init__(self, imgs: np.ndarray, transform=None):
        """
        Args:
            imgs (np.ndarray): Training slices (NumPy array).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.original_imgs = imgs
        self.transform = transform

        # Ensure the input is in the shape [320, 1, 256, 256]
        assert len(self.original_imgs.shape) == 4, "Input images should be 4D: [N, 1, H, W]"

        # Apply the transformations (normalization and flips)
        self.normalized_imgs = np.stack([self._apply_transform(img) for img in self.original_imgs])
        self.horizontal_flip_imgs = np.stack([self._apply_transform(img, horizontal=True) for img in self.original_imgs])
        self.vertical_flip_imgs = np.stack([self._apply_transform(img, vertical=True) for img in self.original_imgs])

        # Concatenate original, horizontal flip, and vertical flip images
        self.augmented_imgs = np.concatenate([self.normalized_imgs, self.horizontal_flip_imgs, self.vertical_flip_imgs], axis=0)

        print(f"Augmented dataset shape: {self.augmented_imgs.shape}")

    def _apply_transform(self, img, horizontal=False, vertical=False):
        """Helper function to apply normalization and optional flips."""
        # The input img is already a NumPy array, so no need to convert to tensor initially
        if horizontal:
            img = np.flip(img, axis=2)  # Flip along width (axis=2)

        if vertical:
            img = np.flip(img, axis=1)  # Flip along height (axis=1)

        # Convert NumPy to Tensor (if needed for normalization)
        img_tensor = torch.from_numpy(img).float()

        # Apply normalization if the transform is defined
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Convert the tensor back to NumPy array for concatenation
        return img_tensor.numpy()

    def __len__(self):
        return len(self.augmented_imgs)

    def __getitem__(self, idx):
        return self.augmented_imgs[idx]

def find_nii_directories(base_dir, modality="FLAIR"):
    """
    Recursively find all directories containing .nii.gz files with specific criteria.

    Args:
        base_dir (str): The base directory to search.

    Returns:
        list: List of directories containing at least one .nii.gz file with the modality.
    """
    nii_directories = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".nii.gz") and modality in file and "cleaned" in file:
                nii_directories.append(os.path.join(root, file))
                break
    return nii_directories

from typing import List, Tuple, Sequence

def load_images(files: List[str], config) -> np.ndarray:
    """Load images from a list of files.
    Args:
        files (List[str]): List of files
        config (Namespace): Configuration
    Returns:
        images (np.ndarray): Numpy array of images
    """
    load_fn = partial(load_nii_nn,
                      slice_range=config.slice_range,
                      size=config.image_size,
                      normalize=config.normalize,
                      equalize_histogram=config.equalize_histogram)
    return load_files_to_ram(files, load_fn)

dirList = ['D:/VascularData/data/nii\\027_S_0074', 'D:/VascularData/data/nii\\018_S_2155', 'D:/VascularData/data/nii\\027_S_2219', 'D:/VascularData/data/nii\\094_S_2238', 'D:/VascularData/data/nii\\127_S_1427', 'D:/VascularData/data/nii\\068_S_2315', 'D:/VascularData/data/nii\\041_S_4037', 'D:/VascularData/data/nii\\099_S_4086', 'D:/VascularData/data/nii\\041_S_4143', 'D:/VascularData/data/nii\\021_S_4254', 'D:/VascularData/data/nii\\137_S_4351', 'D:/VascularData/data/nii\\018_S_4399', 'D:/VascularData/data/nii\\137_S_4466', 'D:/VascularData/data/nii\\006_S_4485', 'D:/VascularData/data/nii\\137_S_4482', 'D:/VascularData/data/nii\\137_S_4536', 'D:/VascularData/data/nii\\021_S_4659', 'D:/VascularData/data/nii\\024_S_4674', 'D:/VascularData/data/nii\\127_S_4765', 'D:/VascularData/data/nii\\116_S_4855', 'D:/VascularData/data/nii\\041_S_4974', 'D:/VascularData/data/nii\\024_S_6005', 'D:/VascularData/data/nii\\941_S_6017', 'D:/VascularData/data/nii\\941_S_6052', 'D:/VascularData/data/nii\\024_S_6202', 'D:/VascularData/data/nii\\035_S_6306', 'D:/VascularData/data/nii\\019_S_6315', 'D:/VascularData/data/nii\\035_S_6380', 'D:/VascularData/data/nii\\341_S_6494', 'D:/VascularData/data/nii\\941_S_6496', 'D:/VascularData/data/nii\\116_S_6458', 'D:/VascularData/data/nii\\941_S_6514', 'D:/VascularData/data/nii\\941_S_6546', 'D:/VascularData/data/nii\\100_S_6493', 'D:/VascularData/data/nii\\941_S_6580', 'D:/VascularData/data/nii\\031_S_6715', 'D:/VascularData/data/nii\\016_S_6708', 'D:/VascularData/data/nii\\126_S_6724', 'D:/VascularData/data/nii\\016_S_6800', 'D:/VascularData/data/nii\\168_S_6860', 'D:/VascularData/data/nii\\137_S_6880', 'D:/VascularData/data/nii\\168_S_6902', 'D:/VascularData/data/nii\\016_S_6939', 'D:/VascularData/data/nii\\126_S_7060', 'D:/VascularData/data/nii\\941_S_10013', 'D:/VascularData/data/nii\\021_S_0337', 'D:/VascularData/data/nii\\011_S_10026', 'D:/VascularData/data/nii\\011_S_6303', 'D:/VascularData/data/nii\\035_S_10068', 'D:/VascularData/data/nii\\126_S_4514', 'D:/VascularData/data/nii\\068_S_0127', 'D:/VascularData/data/nii\\052_S_4944', 'D:/VascularData/data/nii\\126_S_0605', 'D:/VascularData/data/nii\\941_S_10212']
def get_dataloaders(train_base_dir, modality, batch_size=4, transform=None,
                    validation_split=0.1, test_split=0.1, seed=42, config=None, inf=False):
    """
    Prepare and return DataLoaders for training, validation, and testing.

    Args:
        train_base_dir (str): Base directory containing training NIfTI directories.
        batch_size (int, optional): Batch size for DataLoaders. Default is 4.
        transform (callable, optional): Transformations to apply to the data.
        validation_split (float, optional): Fraction of the dataset to use for validation.
        test_split (float, optional): Fraction of the dataset to use for testing.
        seed (int, optional): Random seed for reproducibility.
        config (Namespace, optional): Configuration for loading parameters.

    Returns:
        tuple: (train_loader, validation_loader, test_loader)
    """
    if transform is None:
        transform = transforms.Compose([

            transforms.Normalize((0.5,), (0.5,)),
        ])
    torch.manual_seed(seed)

    print("Data load....")
    train_directories =[]
    for sub_dir in dirList:
        tmp = find_nii_directories(sub_dir, modality)
        for nii_paths in tmp:
            train_directories.append(nii_paths)
        
    print(train_directories)
    #train_directories = [find_nii_directories(base_dir=sub_dir, modality=modality) for sub_dir in dirList]
    if inf is True:
        train_directories = train_directories[:4]
        train_imgs = np.concatenate(load_images(train_directories, config))
        #train_dataset = AugmentedTrainDataset(train_imgs, transform=transform)
        validation_dataset = TrainDataset(train_imgs)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return None, validation_loader, None


    train_imgs = np.concatenate(load_images(train_directories, config))
    #train_dataset = Nifti3DDataset(train_directories, transform=transform, config=config)
    train_dataset =TrainDataset(train_imgs)

    total_size = len(train_dataset)
    validation_size = int(total_size * validation_split)
    test_size = int(total_size * test_split)    
    train_size = total_size - validation_size - test_size

    train_dataset, validation_dataset, test_dataset = random_split(train_dataset, [train_size, validation_size, test_size])

    
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Len train_loader: {len(train_loader)}")
    print(f"Len val_loader: {len(validation_loader)}")
    print(f"Len test_loader: {len(test_loader)}")

    return train_loader, validation_loader, test_loader
