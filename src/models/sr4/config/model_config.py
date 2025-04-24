"""
FastSR - SR4 Model Konfigürasyonu
"""

# Model parametreleri
SR4_MODEL_CONFIG = {
    'scale_factor': 1.875,       # Ölçeklendirme faktörü
    'num_channels': 48,          # Kanal sayısı
    'num_blocks': 14,            # Residual blok sayısı
    'dropout_rate': 0.2          # Dropout oranı
}

# Eğitim parametreleri
SR4_TRAIN_CONFIG = {
    'batch_size': 8,                 # Batch size
    'gradient_accumulation_steps': 4, # Gradient birikimi adımları
    'learning_rate': 0.0001,         # Öğrenme oranı
    'num_epochs': 100,               # Toplam epoch sayısı
    'patience': 30,                  # Erken durma patience değeri
    'max_grad_norm': 1.0             # Gradient clipping için max değer
}

# Veri parametreleri
SR4_DATA_CONFIG = {
    'hr_size': (240, 240),       # Yüksek çözünürlüklü boyut
    'lr_size': (128, 128),       # Düşük çözünürlüklü boyut
    'num_frames': 4              # Video başına işlenecek kare sayısı
}

# Loss parametreleri
SR4_LOSS_CONFIG = {
    'alpha': 0.8,    # L1 loss ağırlığı
    'beta': 0.15,    # Perceptual loss ağırlığı
    'gamma': 0.05    # SSIM loss ağırlığı
} 