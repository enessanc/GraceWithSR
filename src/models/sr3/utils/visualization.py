import matplotlib.pyplot as plt
import torch
import numpy as np

def save_comparison(lr, sr, hr, save_path):
    """
    Düşük çözünürlüklü, süper çözünürlüklü ve yüksek çözünürlüklü görüntüleri karşılaştırmalı olarak kaydeder.
    
    Args:
        lr: Düşük çözünürlüklü görüntü tensor'ı
        sr: Süper çözünürlüklü görüntü tensor'ı
        hr: Yüksek çözünürlüklü görüntü tensor'ı
        save_path: Kaydedilecek dosya yolu
    """
    # Tensor'ları numpy array'e çevir
    lr = lr.squeeze().cpu().numpy().transpose(1, 2, 0)
    sr = sr.squeeze().cpu().numpy().transpose(1, 2, 0)
    hr = hr.squeeze().cpu().numpy().transpose(1, 2, 0)

    # Görüntüleri [0, 1] aralığına normalize et
    lr = np.clip(lr, 0, 1)
    sr = np.clip(sr, 0, 1)
    hr = np.clip(hr, 0, 1)

    # Figure oluştur
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Görüntüleri göster
    axes[0].imshow(lr)
    axes[0].set_title('Low Resolution')
    axes[0].axis('off')
    
    axes[1].imshow(sr)
    axes[1].set_title('Super Resolution')
    axes[1].axis('off')
    
    axes[2].imshow(hr)
    axes[2].set_title('High Resolution')
    axes[2].axis('off')
    
    # Kaydet
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close() 