"""
Configuration parameters for SR2 model
"""

# Model parameters
MODEL_CONFIG = {
    'scale_factor': 1.875,  # 240p -> 128p
    'num_channels': 24,     # Reduced from 32 to 24 for memory efficiency
    'num_blocks': 8,        # Number of residual blocks
    'dropout_rate': 0.1,    # Dropout rate for regularization
}

# Training parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'patience': 10,         # Early stopping patience
    'gamma': 0.1,          # Learning rate decay factor
    'step_size': 5,        # Learning rate decay step
}

# Loss function parameters
LOSS_CONFIG = {
    'alpha': 0.7,          # MSE weight
    'beta': 0.2,           # SSIM weight
    'gamma': 0.1,          # Perceptual weight
}

# Dataset parameters
DATASET_CONFIG = {
    'hr_size': (240, 240),  # High resolution size
    'lr_size': (128, 128),  # Low resolution size
    'num_frames': 4,        # Number of frames per video
}

# Optimizer parameters
OPTIMIZER_CONFIG = {
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 1e-4,
}

# Scheduler parameters
SCHEDULER_CONFIG = {
    'mode': 'min',
    'factor': 0.1,
    'patience': 5,
    'verbose': True,
} 