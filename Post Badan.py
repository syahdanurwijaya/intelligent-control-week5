import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- Pengaturan Kamera ---
cap = cv2.VideoCapture(0)

# Dapatkan resolusi frame
w, h = 640, 480
if cap.isOpened():
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Mengabaikan frame kosong.")
        continue

    # 1. Pra-pemrosesan Gambar
    image = cv2.flip(image, 1) 
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Proses Deteksi Pose MediaPipe
    results = pose.process(image)

    # 3. Penggambaran
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        
        # 4. Menggambar Landmark Pose
        # Menggambar 33 titik landmark dan koneksi kerangka tubuh
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2), # Kuning Sian
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)) # Magenta

        # 5. Mendapatkan koordinat untuk Bounding Box
        
        # Ambil semua koordinat x dan y yang dinormalisasi (0 hingga 1)
        x_coords = [lm.x for lm in results.pose_landmarks.landmark if lm.visibility > 0.5]
        y_coords = [lm.y for lm in results.pose_landmarks.landmark if lm.visibility > 0.5]

        # Pastikan ada landmark yang terdeteksi
        if x_coords and y_coords:
            x_min_norm = min(x_coords)
            x_max_norm = max(x_coords)
            y_min_norm = min(y_coords)
            y_max_norm = max(y_coords)

            # Konversi koordinat normalisasi ke piksel
            xmin_px = int(x_min_norm * w)
            ymin_px = int(y_min_norm * h)
            xmax_px = int(x_max_norm * w)
            ymax_px = int(y_max_norm * h)

            # Tambahkan padding (jarak tepi) agar kotak tidak terlalu ketat
            padding = 30
            xmin_px = max(0, xmin_px - padding)
            ymin_px = max(0, ymin_px - padding)
            xmax_px = min(w, xmax_px + padding)
            # Karena pose bisa saja tidak mencakup kaki, padding di bawah lebih besar
            ymax_px = min(h, ymax_px + padding * 2) 

            # Menggambar Bounding Box
            color_box = (0, 255, 0) # Warna Hijau
            cv2.rectangle(image, (xmin_px, ymin_px), (xmax_px, ymax_px), color_box, 3)
            
            # Menambahkan label "Person"
            cv2.putText(image, "PERSON DETECTED", (xmin_px, ymin_px - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_box, 2)


    # 6. Tampilkan Frame
    cv2.imshow('Person Detection (Pose BBox)', image)
    
    # Keluar jika tombol ESC ditekan
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Bersihkan dan tutup jendela
cap.release()
cv2.destroyAllWindows()