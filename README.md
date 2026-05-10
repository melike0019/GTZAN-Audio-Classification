# GTZAN Müzik Türü Sınıflandırma — Deep CNN Pipeline

Derin Öğrenme dersi final projesi · 2 kişilik takım · GTZAN veri seti üzerinde Baseline CNN ve Derin CNN karşılaştırması.

## Sonuçlar

| Model | Test Accuracy | Parametre | Mimari |
|---|---|---|---|
| Baseline CNN | %66.0 | 42.3M | 2 conv blok + Flatten |
| **Deep CNN (DeepCNNv3)** | **%80.0** | **1.6M** | **4 conv blok + BatchNorm + AdaptiveAvgPool** |

Deep CNN, **26× daha az parametreyle** Baseline'dan **+14 puan** daha yüksek test doğruluğu elde etmiştir.

## Mimari (DeepCNNv3)

- 4 konvolüsyon bloğu: 1 → 64 → 128 → 256 → 512 kanal
- Her blokta BatchNorm + ReLU + MaxPool (son blokta AdaptiveAvgPool)
- Sınıflandırıcı: Linear(512→128) + ReLU + Dropout(0.4) + Linear(128→10)
- Toplam: 1,618,698 parametre

## Eğitim Detayları

- Optimizer: Adam (lr=5e-4)
- Scheduler: Lineer warmup (3 epoch) + LambdaLR (her 15 epoch yarıya)
- Gradient clipping: max_norm=1.0
- Loss: CrossEntropyLoss · Batch size: 16 · Epoch: 40
- Donanım: Google Colab T4 GPU
- Final validation accuracy: %85.3

## Veri Seti

GTZAN — 10 müzik türü × 100 dosya (30 saniye, 22050 Hz). Bilinen bozuk dosya `jazz/jazz.00054.wav` ön işlemede atlandı (toplam 999 örnek).

Train/Val/Test = 699/150/150 (stratified split)

Mel-spectrogram: `n_mels=128, n_fft=2048, hop_length=512`. Z-score normalizasyon train setinin istatistikleriyle uygulandı.

## Proje Yapısı
notebooks/
├── 01_Data_Exploration.ipynb     # Veri keşfi, kulak analizi, spectrogram'lar
├── 02_Dataset_Loader.ipynb       # PyTorch Dataset + DataLoader
├── 03_Deep_CNN_Training.ipynb    # DeepCNNv3 mimarisi + eğitim
├── 04_Test_Evaluation.ipynb      # Test set değerlendirmesi + karşılaştırma
└── ...
src/
├── dataset.py                    # GTZANDataset sınıfı
├── deep_cnn.py                   # DeepCNNv3 mimarisi
└── train_deep.py                 # Eğitim döngüsü
## Kurulum

```bash
git clone https://github.com/melike0019/GTZAN-Audio-Classification.git
cd GTZAN-Audio-Classification
pip install torch torchaudio librosa numpy matplotlib seaborn pandas tqdm gradio
```

## Takım

- Melike Çoldur — Deep CNN mimarisi, eğitim, demo
- Buse Tosuner — Veri ön işleme, Baseline CNN, confusion matrix analizi
