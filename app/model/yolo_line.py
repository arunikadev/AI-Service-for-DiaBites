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

        # YOLO inference
        results = self.model.predict(image_np, conf=0.25, verbose=False)

        result = results[0]
        boxes = result.boxes
        if len(boxes) == 0:
            return []

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )
            # crop per line
            crop = image_np[y1:y2, x1:x2]
            # convert ke PIL
            crop_pil = Image.fromarray(crop)
            detections.append({
                "box": [x1, y1, x2, y2],
                "crop": crop_pil
            })

        # sort atas → bawah
        detections = sorted(
            detections,
            key=lambda d: d["box"][1]
        )
        return detections