import cv2
import numpy as np
from ultralytics import YOLO

class YOLOInference:
    def __init__(self, weight_path: str):
        # Memuat model YOLO
        # Secara otomatis Ultralytics akan mendeteksi apakah environment memiliki GPU atau tidak.
        # Jika tidak, ia akan menggunakan CPU (device='cpu').
        self.model = YOLO(weight_path)

    def predict_and_crop(self, image_bytes: bytes):
        """
        Menerima bytes gambar, mendeteksi bounding box nutrition label,
        dan mengembalikan potongan gambar (cropped image).
        """
        # Ubah bytes menjadi numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Gambar tidak dapat di-decode")

        # Inferensi YOLO
        results = self.model.predict(img, conf=0.25, verbose=False)
        result = results[0]
        boxes = result.boxes
        
        if len(boxes) == 0:
            return None, None # Tidak ada label terdeteksi
            
        # Ambil box dengan confidence tertinggi (asumsi ini adalah tabel label gizi)
        best_box = max(boxes, key=lambda x: x.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        
        # Crop gambar
        cropped_img = img[y1:y2, x1:x2]
        
        return cropped_img, (x1, y1, x2, y2)
