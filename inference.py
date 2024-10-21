import argparse
from Data.dataload import get_dataloaders
from models.models import FeatureReconstructor
from utils.pytorch_ssim import SSIMLoss  
from analysis.vis import *
import torch
import torch.nn as nn
import torch.optim as optim
import os

#D:/Download/Downloads/nii
#D:/VascularData/data/nii
#D:/Download/abnormal.csv
def parse_args():
    parser = argparse.ArgumentParser(description='Train Feature Autoencoder on 3D DICOM Images')
    parser.add_argument('--csv_path', type=str, default="D:/Download/abnormal.csv", help='Path to the CSV file containing DICOM paths and labels')
    parser.add_argument('--train_base_dir', type=str, default="D:/Download/Downloads/nii", help='Base directory for training DICOM files')
    parser.add_argument('--modality', type=str, default="FLAIR", help='Data modality')
    parser.add_argument('--batch_size', type=int, default=32 , help='Batch size for DataLoaders')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')

    parser.add_argument('--image_size', type=int,
                         default=128, help='Image size')
    parser.add_argument('--slice_range', type=int,
                            nargs='+', default=(65, 145), help='Slice range')
    parser.add_argument('--normalize', type=bool,
                            default=False, help='Normalize images between 0 and 1')
    parser.add_argument('--equalize_histogram', type=bool,
                            default=True, help='Equalize histogram')

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    return parser.parse_args()

def visualize_volume(volumes, num_slices=5):
    B, _, H, W = volumes.shape  # Adjust to match the shape [batch size, 1, 128, 128]
    
    # Take the first volume from the batch (index 0)
    volume = volumes[0, 0].cpu().numpy()  # Shape: [128, 128]
    
    # Choose the step to evenly sample slices from the height (H)
    slice_step = max(1, H // num_slices)
    
    plt.imshow(volume, cmap="gray")  # No need for volume[slice_idx] since it's already 2D
    plt.axis('off')
    plt.show()

def main():
    args = parse_args()

    train_loader, validation_loader, test_loader = get_dataloaders(
        train_base_dir=args.train_base_dir,
        csv_path=args.csv_path,
        modality=args.modality,
        batch_size=args.batch_size,
        transform=None,  
        validation_split=0.1,
        seed=42,
        config=args,
        inf=True
    )

    from argparse import Namespace
    config = Namespace()
    config.image_size = 128
    config.hidden_dims = [100, 150, 200, 300]
    config.generator_hidden_dims = [300, 200, 150, 100]
    config.discriminator_hidden_dims = [100, 150, 200, 300]
    config.dropout = 0.2
    config.extractor_cnn_layers = ['layer1', 'layer2']
    config.keep_feature_prop = 1.0
    config.random_extractor = False
    config.loss_fn = 'ssim'

    model = FeatureReconstructor(config).to(args.device)

    print("Feature Autoencoder Encoder:")
    print(model.ae.enc)
    print("\nFeature Autoencoder Decoder:")
    print(model.ae.dec)
    model.load("./saved_models/best_model.pth")
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for volumes in validation_loader:
            volumes = volumes.to(args.device)
            #B, C, D, H, W = volumes.shape
            #volumes_slices = volumes.view(B * D, C, H, W)
            anomaly_map, anomaly_score = model.predict_anomaly(volumes)
            print(anomaly_map.shape)
            print(anomaly_score)
            loss_dict = model.loss(volumes)
            loss = loss_dict['rec_loss']
            val_loss += loss.item()
            plot_anomaly_map_with_original(volumes, anomaly_map, anomaly_score, -0.22)


if __name__ == '__main__':
    main()
