# fae/main.py

import argparse
from Data.dataload import get_dataloaders
from models.models import FeatureReconstructor
from utils.pytorch_ssim import SSIMLoss  
from analysis.vis import *
import torch
import torch.nn as nn
import torch.optim as optim
import os
#/home/kbh/Downloads/csv/abnormal.csv
#/home/kbh/Downloads/nii
#D:/VascularData/data/nii
#D:/Download/Downloads/nii
def parse_args():
    parser = argparse.ArgumentParser(description='Train Feature Autoencoder on 3D DICOM Images')
    parser.add_argument('--csv_path', type=str, default="/home/kbh/Downloads/csv/abnormal.csv", help='Path to the CSV file containing DICOM paths and labels')
    parser.add_argument('--train_base_dir', type=str, default="/home/kbh/Downloads/nii", help='Base directory for training DICOM files')
    parser.add_argument('--modality', type=str, default="FLAIR", help='Data modality')
    parser.add_argument('--batch_size', type=int, default=32 , help='Batch size for DataLoaders')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')

    parser.add_argument('--image_size', type=int,
                         default=256, help='Image size')
    parser.add_argument('--slice_range', type=int,
                            nargs='+', default=(65, 145), help='Slice range')
    parser.add_argument('--normalize', type=bool,
                            default=False, help='Normalize images between 0 and 1')
    parser.add_argument('--equalize_histogram', type=bool,
                            default=True, help='Equalize histogram')

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    return parser.parse_args()


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
        inf=False,
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_val_loss = float('inf') 

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, volumes in enumerate(train_loader):
            optimizer.zero_grad()
            volumes = volumes.to(args.device)  # Shape: (B, 1, H, W)
            #print(volumes.shape)
            #visualize_volume(volumes, num_slices=5)
            

            # B, H, W, D = volumes.shape
            # volumes_slices = volumes.view(B*D, 1, H, W)  # Shape: (B*D, 1, H, W)
            # print(volumes_slices.shape)
            # Forward pass
            loss_dict = model.loss(volumes)
            loss = loss_dict['rec_loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{args.epochs}], Average Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for volumes in validation_loader:
                volumes = volumes.to(args.device)
                #B, C, D, H, W = volumes.shape
                #volumes_slices = volumes.view(B * D, C, H, W)
                #anomaly_map, anomaly_score = model.predict_anomaly(volumes)

                loss_dict = model.loss(volumes)
                loss = loss_dict['rec_loss']
                val_loss += loss.item()
                #plot_anomaly_map_with_original(volumes, anomaly_map, slice_idx=0)


        val_loss /= len(validation_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss  
            os.makedirs(args.save_dir, exist_ok=True)
            best_model_path = os.path.join(
                args.save_dir, 
                f"best_model_epoch_{epoch}_batchsize_{args.batch_size}_lr_{args.lr}.pth"
            )
            model.save(config, f"best_model.pth", directory=args.save_dir)
            print(f"New best model saved to {best_model_path}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for volumes in test_loader:
            volumes = volumes.to(args.device)
            # B, C, D, H, W = volumes.shape
            # volumes_slices = volumes.view(B * D, C, H, W)

            loss_dict = model.loss(volumes)
            loss = loss_dict['rec_loss']
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    final_model_path = os.path.join(args.save_dir, "model_final.pth")
    model.save(config, "model_final.pth", directory=args.save_dir)
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    main()
