# GraceWithSR: Video İyileştirme ve Süper Çözünürlük İşlem Hattı

## Proje Hakkında

Bu proje, düşük kaliteli videoları iyileştirmek için Grace ve SR3 modellerini birleştiren bir video iyileştirme işlem hattı uygular. Proje, ELE567 dersi kapsamında geliştirilmiş olup, video kalitesini artırmak için modern derin öğrenme tekniklerini kullanmaktadır.

### Süper Çözünürlük Modelleri

Projede dört farklı süper çözünürlük modeli geliştirilmiştir:

1. **SR1 Modeli** ([Detaylar](src/models/sr1/README.md))
   - Temel süper çözünürlük modeli
   - Hafif mimari (223K parametre)
   - Frame bazlı işleme
   - Residual learning ve pixel shuffle tabanlı upsampling

2. **SR2 Modeli** ([Detaylar](src/models/sr2/README.md))
   - SR1'in geliştirilmiş versiyonu
   - SE (Squeeze-and-Excitation) katmanları
   - Gelişmiş loss fonksiyonları (L1, Perceptual, SSIM)
   - Batch Normalization ve Dropout

3. **SR3 Modeli** ([Detaylar](src/models/sr3/README.md))
   - SR2'nin geliştirilmiş versiyonu
   - Channel ve Spatial Attention mekanizmaları
   - Gelişmiş memory optimizasyonları
   - Lion Optimizer kullanımı

4. **SR4 Modeli** ([Detaylar](src/models/sr4/README.md))
   - SR3'ün geliştirilmiş versiyonu
   - Metin tanıma için özel Text Attention katmanları
   - Gelişmiş dikkat mekanizmaları
   - AdamW Optimizer kullanımı

### Temel Özellikler

- 240p çözünürlüğündeki giriş videolarını iyileştirme
- Grace modeli ile video kalitesini artırma
- SR3 modeli ile süper çözünürlük uygulama
- Google Colab ve yerel ortam desteği
- CUDA GPU hızlandırma desteği

### İşlem Hattı

1. 240p çözünürlüğünde bir giriş videosu alır
2. 128p'ye küçültür
3. Grace modeli ile işler (128p üzerinde eğitilmiş)
4. SR3 modeli kullanarak tekrar 240p'ye yükseltir

## Proje Raporu

Detaylı proje bilgileri ve teknik açıklamalar için [ELE567ProjeRapor.pdf](ELE567ProjeRapor.pdf) dosyasını inceleyebilirsiniz.

## Proje Yapısı

```
GraceWithSR/
├── src/                    # Kaynak kodlar
│   ├── integration/        # Entegrasyon kodları
│   │   ├── pipeline/      # İşlem hattı kodları
│   │   └── utils/         # Yardımcı fonksiyonlar
│   ├── utils/             # Genel yardımcı fonksiyonlar
│   └── models/            # Model tanımları
│       ├── sr1/          # SR1 modeli
│       ├── sr2/          # SR2 modeli
│       ├── sr3/          # SR3 modeli
│       └── sr4/          # SR4 modeli
├── external/              # Harici bağımlılıklar
│   └── Grace/            # Grace modeli ve kütüphanesi
├── models/                # Eğitilmiş modeller
│   └── SR3Model/         # SR3 model dosyaları
├── dataset/              # Veri seti klasörü
│   ├── input/           # Giriş videoları
│   └── output/          # Çıkış videoları
├── results/              # İşlem sonuçları
├── setup_colab.sh       # Colab kurulum betiği
├── run.sh               # Çalıştırma betiği
└── requirements.txt     # Python bağımlılıkları
```

## Gereksinimler

- Python 3.7+
- CUDA destekli GPU (önerilen)
- FFmpeg
- Git
- Google Colab hesabı (Colab için)

## Kurulum

### Google Colab için

1. Depoyu Colab'a klonlayın
2. Kurulum betiğini çalıştırın:
```bash
chmod +x setup_colab.sh
./setup_colab.sh
```

### Yerel Ortam için

1. Depoyu klonlayın:
```bash
git clone https://github.com/enessanc/GraceWithSR.git
cd GraceWithSR
```

2. Grace alt modülünü başlatın:
```bash
git submodule update --init --recursive
```

3. Python sanal ortamı oluşturun ve bağımlılıkları yükleyin:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Kullanım

1. Giriş videonuzu `dataset/input/` klasörüne yerleştirin

2. İşlem hattını çalıştırın:
```bash
chmod +x run.sh
./run.sh
```

İşlenmiş video `dataset/output/` klasörüne kaydedilecektir

### Video Karşılaştırma

İşlenmiş ve orijinal videoları karşılaştırmak için:
```bash
python dataset/compare_video_frames.py input_video.mp4 output_video.mp4
```

## Veri Seti

Örnek videolar ve veri setleri için [dataset/README.md](dataset/README.md) dosyasındaki linki kullanabilirsiniz.

## Notlar

- İşlem hattı optimal performans için CUDA gerektirir
- Google Colab ortamında test edilmiştir
- Giriş videoları 240p çözünürlüğünde olmalıdır
- Grace modeli 128p görüntüler üzerinde eğitilmiştir
- SR3 modeli 128p'den 240p'ye yükseltme için eğitilmiştir

## İletişim

Proje ile ilgili sorularınız için:
- E-posta: [enes.sancak2001@gmail.com](mailto:enes.sancak2001@email.com)
