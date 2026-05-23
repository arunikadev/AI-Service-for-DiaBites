import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if base_dir not in sys.path:
    sys.path.append(base_dir)

from model.yolo_line import YOLOInference

root_dir = os.path.dirname(base_dir)
weights_dir = os.path.join(root_dir, "weights")
print("Loading YOLO model...")

try:
    yolo_model = YOLOInference(
        os.path.join(weights_dir, "GS.pt")
    )
    print("[OK] YOLO model berhasil di-load!")

except Exception as e:
    print("[ERROR] Error loading model:", e)
    sys.exit(1)

image_filename = "c.jpg"

image_path = os.path.join(
    os.path.dirname(__file__),
    image_filename
)

if not os.path.exists(image_path):
    print(f"[ERROR] Gambar tidak ditemukan: {image_path}")
    sys.exit(1)

print(f"Memproses gambar: {image_path}")

image = Image.open(image_path).convert("L").convert("RGB")

image_np = np.array(image)

detections = yolo_model.predict_lines(image_np)

if not detections:
    print("[ERROR] Tidak ada line terdeteksi.")
    sys.exit(1)

print(f"[OK] Jumlah line terdeteksi: {len(detections)}")

detections = sorted(
    detections,
    key=lambda x: x["box"][1]
)

fig, ax = plt.subplots(
    1,
    figsize=(14, 10)
)

ax.imshow(image_np)

for idx, det in enumerate(detections):
    x1, y1, x2, y2 = det["box"]
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle(
        (x1, y1),
        width,
        height,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )

    ax.add_patch(rect)
    ax.text(
        x1,
        y1 - 5,
        f"Line {idx+1}",
        color='red',
        fontsize=10,
        weight='bold'
    )

    print(f"Line {idx+1}: {[x1, y1, x2, y2]}")

plt.axis("off")
plt.title("YOLO Line Detection")
plt.show()

output_dir = os.path.join(
    os.path.dirname(__file__),
    "outputs"
)

os.makedirs(output_dir, exist_ok=True)

for idx, det in enumerate(detections):
    crop_img = det["crop"]
    save_path = os.path.join(
        output_dir,
        f"line_{idx+1}.jpg"
    )
    crop_img.save(save_path)
    print(f"[OK] Crop disimpan: {save_path}")

for idx, det in enumerate(detections):
    crop_img = det["crop"]
    plt.figure(figsize=(12, 2))
    plt.imshow(crop_img)
    plt.title(f"Crop Line {idx+1}")
    plt.axis("off")
    plt.show()