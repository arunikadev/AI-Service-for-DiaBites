import numpy as np
from PIL import Image
from ultralytics import YOLO


class YOLOInference:

    def __init__(self, weight_path: str):
        # Memuat model YOLO
        # Secara otomatis Ultralytics akan mendeteksi apakah environment memiliki GPU atau tidak.
        # Jika tidak, ia akan menggunakan CPU (device='cpu').
        self.model = YOLO(weight_path)

    def predict_lines(self, image_np):
        """
        Detect setiap line text menggunakan YOLO.

        Parameters
        ----------
        image_np : np.ndarray
            Gambar format numpy array (H, W, C)

        Returns
        -------
        list
            List hasil detection:
            [
                {
                    "box": [x1, y1, x2, y2],
                    "crop": PIL.Image
                }
            ]
        """

        if image_np is None:
            raise ValueError("Image kosong")

        # YOLO inference — conf=0.1 supaya baris dengan confidence rendah (Gula) tetap terdeteksi
        results = self.model.predict(image_np, conf=0.1, verbose=False)

        result = results[0]
        boxes = result.boxes
        if len(boxes) == 0:
            return []

        img_h, img_w = image_np.shape[:2]

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Extend x2 untuk menangkap kolom nilai (mis. "100 kkal") yang sering di luar bbox.
            # Offset PROPORSIONAL terhadap lebar bbox (bukan piksel tetap) supaya
            # resolution-independent: gambar besar/kecil sama-sama benar. Fixed +px gagal
            # di gambar kecil (mis. 219px) — menarik kolom % AKG → "100"+"7%"→"107".
            # 18% teruji 4/4 baris benar pada gambar 512px maupun 219px.
            box_w = x2 - x1
            x2_extended = min(x2 + int(box_w * 0.18), img_w)

            crop = image_np[y1:y2, x1:x2_extended]
            crop_pil = Image.fromarray(crop)
            detections.append({
                "box": [x1, y1, x2, y2],   # box asli YOLO (untuk referensi)
                "crop": crop_pil            # crop diperlebar ke kanan
            })

        # sort atas → bawah
        detections = sorted(
            detections,
            key=lambda d: d["box"][1]
        )
        return detections