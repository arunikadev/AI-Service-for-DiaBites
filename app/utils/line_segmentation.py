import cv2
import numpy as np

def segment_lines(image):
    """
    Memotong gambar tabel gizi (hasil crop YOLO) menjadi baris-baris horizontal.
    Menggunakan teknik Horizontal Projection Profile untuk mencari baris teks.
    """
    if image is None:
        return []
        
    # 1. Konversi ke Grayscale dan Binarize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Cek polaritas warna (apakah teks hitam di background putih atau sebaliknya)
    # Asumsinya: background memiliki area yang jauh lebih luas daripada teks.
    white_pixels = np.sum(thresh == 255)
    black_pixels = np.sum(thresh == 0)
    
    is_dark_bg = False
    # Projection profile membutuhkan teks berwarna putih (255) dan background hitam (0)
    if white_pixels > black_pixels:
        thresh = cv2.bitwise_not(thresh)
    else:
        is_dark_bg = True
        
    # Jika background gelap (teks terang), kita INVERT GAMBAR ASLI 
    # karena model CRNN dilatih khusus untuk teks gelap di atas background terang.
    if is_dark_bg:
        image = 255 - image
        
    # Hapus garis tabel (horizontal dan vertikal) yang bisa merusak projection profile
    # 1. Garis Horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, 0, -1) # Hapus garis dengan mewarnainya hitam

    # 2. Garis Vertikal
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, 0, -1) # Hapus garis dengan mewarnainya hitam
    
    # 2. Horizontal Projection (Hitung jumlah pixel teks per baris y)
    proj = np.sum(thresh, axis=1)
    
    # 3. Cari puncak dari proyeksi (baris yang mengandung teks)
    threshold_val = np.max(proj) * 0.05  # Filter noise ringan
    in_line = False
    lines_y = []
    start_y = 0
    
    for y, val in enumerate(proj):
        if not in_line and val > threshold_val:
            in_line = True
            start_y = y
        elif in_line and val <= threshold_val:
            in_line = False
            # Tambahkan padding atas bawah
            pad = 5
            y1 = max(0, start_y - pad)
            y2 = min(image.shape[0], y + pad)
            
            # Abaikan noise yang terlalu tipis (misal garis tepi tabel)
            if y2 - y1 > 15:
                lines_y.append((y1, y2))
                
    # 4. Potong gambar asli selebar gambar untuk setiap baris
    cropped_lines = []
    for y1, y2 in lines_y:
        # Kita ambil selebar gambar penuh (axis x tidak dipotong)
        line_img = image[y1:y2, :]
        cropped_lines.append(line_img)
        
    return cropped_lines
