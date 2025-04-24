"""
Model ve eğitim parametreleri için konfigürasyon dosyası.
"""

# Model parametreleri
MODEL_CONFIG = {
    'scale_factor': 1.875,
    'num_channels': 48,
    'num_blocks': 10,
    'dropout_rate': 0.2
}

# Veri seti parametreleri
DATASET_CONFIG = {
    'hr_size': (240, 240),
    'lr_size': (128, 128),
    'num_frames': 4
}

# Eğitim parametreleri
TRAIN_CONFIG = {
    'batch_size': 4,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_epochs': 100,
    'patience': 10,
    'checkpoint_dir': 'checkpoints'
}

# Loss parametreleri
LOSS_CONFIG = {
    'alpha': 0.7,  # L1 loss ağırlığı
    'beta': 0.2,   # SSIM loss ağırlığı
    'gamma': 0.1   # Perceptual loss ağırlığı
}

# Test parametreleri
TEST_CONFIG = {
    'batch_size': 1,
    'num_workers': 4,
    'output_dir': 'test_results'
} 