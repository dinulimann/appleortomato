import cv2
import numpy as np
from tensorflow import keras
import os

# Load model
MODEL_PATH = 'apple_tomato_model.keras'

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'apple_tomato_model_final.keras'

if not os.path.exists(MODEL_PATH):
    print("Error: Model tidak ditemukan!")
    print("Silakan jalankan train_model.py terlebih dahulu.")
    exit(1)

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load class labels
class_labels = {}
if os.path.exists('class_labels.txt'):
    with open('class_labels.txt', 'r') as f:
        for line in f:
            idx, name = line.strip().split(':')
            class_labels[int(idx)] = name
else:
    # Default labels jika file tidak ada
    class_labels = {0: 'apples', 1: 'tomatoes'}

print(f"Class labels: {class_labels}")

# Inisialisasi kamera
print("Membuka kamera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

IMG_SIZE = (224, 224)
confidence_threshold = 0.6

print("\n=== INSTRUKSI ===")
print("- Arahkan kamera ke apel atau tomat")
print("- Tekan 'q' untuk keluar")
print("- Tekan 's' untuk screenshot")
print("================\n")

frame_count = 0
prediction_text = ""
confidence = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera!")
        break

    # Mirror frame untuk pengalaman lebih natural
    frame = cv2.flip(frame, 1)

    # Proses prediksi setiap 5 frame untuk performa lebih baik
    if frame_count % 5 == 0:
        # Ambil region of interest di tengah frame
        height, width = frame.shape[:2]
        roi_size = min(height, width) - 100
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2

        roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]

        # Preprocess image
        img = cv2.resize(roi, IMG_SIZE)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediksi
        prediction = model.predict(img, verbose=0)[0][0]

        # Tentukan class dan confidence
        if prediction > 0.5:
            class_idx = 1
            confidence = prediction
        else:
            class_idx = 0
            confidence = 1 - prediction

        class_name = class_labels.get(class_idx, 'Unknown')

        # Format output
        if confidence > confidence_threshold:
            prediction_text = f"{class_name.upper()}"
            if confidence > 0.9:
                color = (0, 255, 0)  # Green untuk confidence tinggi
            else:
                color = (0, 255, 255)  # Yellow untuk confidence sedang
        else:
            prediction_text = "TIDAK YAKIN"
            color = (0, 0, 255)  # Red
            confidence = confidence

        # Gambar ROI box
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), color, 3)

    # Display prediction
    # Background untuk text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (600, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Text
    cv2.putText(frame, prediction_text, (20, 60),
                cv2.FONT_HERSHEY_BOLD, 1.5, color, 3)
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Instruksi
    cv2.putText(frame, "Press 'q' to quit | 's' to screenshot", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Tampilkan frame
    cv2.imshow('Apple or Tomato Classifier', frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Keluar dari aplikasi...")
        break
    elif key == ord('s'):
        screenshot_name = f"screenshot_{frame_count}.jpg"
        cv2.imwrite(screenshot_name, frame)
        print(f"Screenshot disimpan: {screenshot_name}")

    frame_count += 1

