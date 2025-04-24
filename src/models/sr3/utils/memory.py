import torch
import gc

def setup_cuda():
    # Tüm CUDA cihazlarını temizle
    torch.cuda.empty_cache()
    gc.collect()

    # CUDA memory ayarları
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def print_memory_status(prefix=""):
    print(f"{prefix}GPU Memory Durumu:")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Cached Memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB") 