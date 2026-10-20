# Apple or Tomato Image Classifier

Aplikasi klasifikasi gambar untuk mendeteksi apel atau tomat secara real-time menggunakan kamera.

## Overview (English)

Real-time image classifier that tells **apples from tomatoes** through a webcam.

- **Model:** transfer learning on **MobileNetV2** (ImageNet weights), 224×224 input, binary classification, with data augmentation. The best checkpoint is saved during training, together with a training-history chart.
- **App:** OpenCV camera window with a region-of-interest box, live prediction and confidence; `s` saves a screenshot, `q` quits.
- **Data:** [Apples or Tomatoes](https://www.kaggle.com/datasets/samuelcortinhas/apples-or-tomatoes-image-classification) dataset from Kaggle, downloaded by `download_dataset.py`.

```bash
pip install -r requirements.txt
python download_dataset.py && python train_model.py && python camera_app.py
```

Detailed instructions below are in Indonesian.

---


## Dataset

Dataset diambil dari Kaggle: [Apples or Tomatoes Image Classification](https://www.kaggle.com/datasets/samuelcortinhas/apples-or-tomatoes-image-classification)

## Instalasi

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download dataset (sudah otomatis):
```bash
python download_dataset.py
```

## Cara Menggunakan

### 1. Training Model

Jalankan script training untuk membuat model:

```bash
python train_model.py
```

Script ini akan:
- Load dan augment dataset
- Training model menggunakan MobileNetV2 (transfer learning)
- Menyimpan best model sebagai `apple_tomato_model.keras`
- Membuat grafik training history (`training_history.png`)
- Menyimpan class labels (`class_labels.txt`)

Training biasanya memakan waktu 10-20 menit tergantung spesifikasi komputer.

### 2. Menjalankan Aplikasi Kamera

Setelah model selesai training, jalankan aplikasi kamera:

```bash
python camera_app.py
```

**Kontrol:**
- `q` - Keluar dari aplikasi
- `s` - Screenshot hasil prediksi

**Tips:**
- Pastikan objek (apel/tomat) berada di dalam kotak hijau/kuning
- Gunakan pencahayaan yang baik
- Hindari background yang terlalu ramai

## Struktur Project

```
.
├── README.md                          # File ini
├── requirements.txt                   # Dependencies
├── download_dataset.py                # Script download dataset
├── train_model.py                     # Script training model
├── camera_app.py                      # Aplikasi kamera
├── apple_tomato_model.keras          # Model terbaik (hasil training)
├── apple_tomato_model_final.keras    # Model final (hasil training)
├── training_history.png              # Grafik training (hasil training)
└── class_labels.txt                  # Label kelas (hasil training)
```

## Spesifikasi Model

- **Base Model:** MobileNetV2 (pretrained pada ImageNet)
- **Input Size:** 224x224 pixels
- **Classes:** Binary classification (Apples vs Tomatoes)
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy

## Hasil Training

Setelah training selesai, Anda akan mendapatkan:
- Accuracy pada test set (ditampilkan di console)
- Grafik training history yang menunjukkan accuracy dan loss
- Model yang siap digunakan untuk prediksi

## Troubleshooting

**Kamera tidak terbuka:**
- Pastikan kamera tidak digunakan aplikasi lain
- Periksa permission kamera di sistem operasi Anda

**Model tidak ditemukan:**
- Pastikan sudah menjalankan `train_model.py` terlebih dahulu
- Periksa apakah file `apple_tomato_model.keras` ada

**Low confidence predictions:**
- Pastikan pencahayaan cukup
- Objek harus jelas terlihat dan berada di dalam ROI box
- Coba dekatkan objek ke kamera

## Requirements

- Python 3.8+
- Webcam/Camera
- Minimal 4GB RAM
- GPU (optional, untuk training lebih cepat)