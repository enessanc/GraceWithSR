import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from google.colab import drive

from network.sr1_model import FastSR
from processing.dataset import ProcessedVideoDataset
from utils.trainer import train_with_validation
from config.config import model_params, train_params, dataset_params

def main():
    # Google Drive'ı bağla
    drive.mount('/content/drive')
    os.chdir('/content/drive/MyDrive/Colab Notebooks')

    # Cihaz
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    processed_dir = "VideoDataSet"
    output_dir = "results"

    # Veri setlerini yükle
    print("Veri setleri yükleniyor...")
    video_paths = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir)
                  if f.endswith(('.mp4', '.avi', '.mov'))]

    train_paths, temp_paths = train_test_split(video_paths, train_size=0.7, random_state=42)
    val_paths, test_paths = train_test_split(temp_paths, train_size=0.5, random_state=42)

    train_dataset = ProcessedVideoDataset(train_paths, **dataset_params)
    val_dataset = ProcessedVideoDataset(val_paths, **dataset_params)
    test_dataset = ProcessedVideoDataset(test_paths, **dataset_params)

    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'],
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'],
                          shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=train_params['batch_size'],
                           shuffle=False, num_workers=4)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Modeli oluştur ve eğit
    print("Model eğitiliyor...")
    model = FastSR(**model_params).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])

    train_losses, val_losses = train_with_validation(
        model, train_loader, val_loader, criterion, optimizer,
        device, train_params['num_epochs'], train_params['patience']
    )

    # Eğitim sonuçlarını görselleştir
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()

if __name__ == "__main__":
    main()
