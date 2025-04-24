# GraceWithSR: Video İyileştirme İşlem Hattı

Bu proje, video kalitesini iyileştirmek için Grace ve SR3 modellerini birleştiren bir video iyileştirme işlem hattı uygular. İşlem hattı:
1. 240p çözünürlüğünde bir giriş videosu alır
2. 128p'ye küçültür
3. Grace modeli ile işler (128p üzerinde eğitilmiş)
4. SR3 modeli kullanarak tekrar 240p'ye yükseltir

## Gereksinimler

- Python 3.7+
- CUDA destekli GPU
- FFmpeg
- Git

## Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/enessanc/GraceWithSR.git
cd GraceWithSR
```

2. Kurulum betiğini çalıştırın:
```bash
chmod +x setup.sh
./setup.sh
```

Bu işlem:
- Sistem bağımlılıklarını yükler
- Python sanal ortamı oluşturur
- CUDA desteği ile PyTorch'u yükler
- Grace ve SR3 bağımlılıklarını yükler
- Gerekli modelleri indirir

## Kullanım

1. Giriş videonuzu `dataset/input/input_video.mp4` konumuna yerleştirin

2. İşlem hattını çalıştırın:
```bash
chmod +x run.sh
./run.sh
```

İşlenmiş video `dataset/output/output_video.mp4` konumuna kaydedilecektir

## Proje Yapısı

```
GraceWithSR/
├── src/
│   └── integration/
│       └── pipeline/
│           └── grace_sr3_pipeline.py
├── external/
│   ├── Grace/
│   └── SR3/
├── models/
│   ├── Grace/
│   └── SR3/
├── dataset/
│   ├── input/
│   └── output/
├── setup.sh
├── run.sh
└── requirements.txt
```

## Notlar

- İşlem hattı optimal performans için CUDA gerektirir ve Colab ortamında test edilmiştir. Her ortamda çalışmayabilir.
