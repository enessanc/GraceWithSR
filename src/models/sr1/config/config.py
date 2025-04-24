# Model parametreleri
model_params = {
    'scale_factor': 4,
    'num_channels': 32
}

# Eğitim parametreleri
train_params = {
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'patience': 10
}

# Veri seti parametreleri
dataset_params = {
    'target_size': (64, 64),
    'num_frames': 30
} 