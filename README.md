# DiaBites AI-Service 

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-Machine%20Learning-lightgrey)

**DiaBites AI-Service** adalah *backend* layanan kecerdasan buatan (AI) terintegrasi yang dirancang untuk menganalisis informasi nilai gizi (Nutrition Facts) dari kemasan makanan/minuman, serta memberikan rekomendasi konsumsi yang dipersonalisasi bagi penderita atau individu yang berisiko diabetes.

Aplikasi ini menggunakan perpaduan **Computer Vision** (YOLO), **Optical Character Recognition** (CRNN), dan **Machine Learning** (LightGBM) untuk memproses gambar menjadi sebuah *insight* kesehatan yang berharga.

---

## Fitur Utama

- **Pendeteksi Baris Teks Gizi (YOLO)**: Mendeteksi dan memotong baris-baris informasi gizi dari sebuah gambar label kemasan secara akurat.
- **Ekstraksi Teks (OCR CRNN)**: Membaca teks dari potongan gambar hasil YOLO untuk mengenali angka dan jenis nutrisi.
- **Smart Nutrition Parser**: Menggunakan algoritma *RegEx* dan *Fuzzy Matching* untuk mengenali nilai Kalori, Karbohidrat, Gula, Lemak, dan Natrium/Sodium secara otomatis dari sekumpulan teks acak.
- **Rekomendasi Cerdas (LightGBM)**: Memberikan klasifikasi rekomendasi berdasarkan profil kesehatan pengguna (Kategori Usia, BMI, Tipe Diabetes).
- **Fast & Async API**: Dibangun di atas FastAPI untuk menjamin performa respons yang tinggi dan penanganan *request* secara asinkron.

---

## Tech Stack

* **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/), Uvicorn
* **Computer Vision & OCR**: OpenCV, Pillow, Ultralytics (YOLOv8), TensorFlow / Keras
* **Machine Learning**: Scikit-Learn, LightGBM, Pandas, Joblib
* **Utility**: FuzzyWuzzy, python-Levenshtein

---

## Struktur Proyek

```text
AI-Service/
├── app/
│   ├── main.py                # Entry point FastAPI & Definisi Endpoint Utama
│   ├── model/                 # Logika pembungkus (Wrapper) untuk model AI (YOLO, CRNN, LightGBM)
│   ├── utils/                 # Utilities (Nutrition Parser, dll)
│   └── test/                  # Kumpulan script pengujian lokal (Testing script)
├── weights/                   # (Wajib diisi) Folder penyimpanan file model weights
│   ├── 29m1.pt                # Model weights untuk YOLO
│   ├── crnn_model-GS.keras    # Model weights untuk CRNN (OCR)
│   ├── vocab.json             # Kosakata (Vocabulary) pemetaan karakter untuk CRNN
│   └── model1.pkl             # Model ML klasifikasi rekomendasi (LightGBM)
├── ocr-service/               # Direktori Virtual Environment Python (Opsional)
├── requirements.txt           # Daftar dependensi library Python
└── README.md                  # Dokumentasi proyek ini
```

*(Catatan: File di dalam folder `weights/` memiliki ukuran besar dan biasanya tidak diunggah ke repositori Git. Pastikan file model tersebut sudah dimasukkan ke foldernya sebelum menjalankan server.)*

---

## Panduan Menjalankan (Getting Started)

### 1. Prasyarat (Prerequisites)
- Python 3.9 atau lebih baru.
- Virtual Environment (sangat direkomendasikan).

### 2. Instalasi
*Clone* repositori ini, kemudian masuk ke direktori proyek:
```bash
git clone https://github.com/arunikadev/AI-Service-for-DiaBites.git
cd AI-Service-for-DiaBites
```

Aktifkan *virtual environment* Anda (contoh menggunakan `venv` bawaan Python):
```bash
python -m venv ocr-service
# Untuk Windows:
ocr-service\Scripts\activate
# Untuk Mac/Linux:
source ocr-service/bin/activate
```

Instal semua dependensi yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Server Lokal
Pastikan semua file model telah berada di folder `weights/`. Jalankan server FastAPI dengan Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Server akan berjalan secara default di `http://127.0.0.1:8000`.

---

## Dokumentasi API

FastAPI menyediakan dokumentasi UI interaktif secara otomatis. Setelah server berjalan, Anda dapat menjelajahi seluruh endpoint melalui browser:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

### Endpoint Spesifikasi

#### 1. Health Check
Memeriksa status kesiapan *service* dan memastikan semua arsitektur model telah berhasil dimuat ke dalam memori.
- **URL**: `/health`
- **Method**: `GET`
- **Response**:
```json
{
  "status": "ready",
  "models": {
    "yolo": true,
    "crnn": true,
    "ml": true
  }
}
```

#### 2. Predict Nutrition & Recommendation
Menerima gambar Nutrition Facts, mengekstrak nilai gizi, dan mengembalikan hasil analisis serta rekomendasi kecocokan konsumsi.
- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Data**:
  - `image` (File): Gambar label informasi nilai gizi (format JPG, JPEG, PNG).
  - `age_group` (String): Kelompok umur pengguna (misal: `adult`, `elderly`).
  - `bmi_category` (String): Kategori berat badan/BMI pengguna (misal: `normal`, `overweight`, `obese`).
  - `diabetes_type` (String): Tipe/Kondisi diabetes pengguna (misal: `type1`, `type2`, `none`).

- **Success Response (200 OK)**:
```json
{
  "status": "success",
  "lines_detected": 6,
  "raw_ocr_text": "Energi Total 60 kkal Lemak Total 0 g Karbohidrat Total 21 g Gula 0 g Garam (Natrium) 70 mg",
  "detected_nutrition": {
    "calories": 60.0,
    "fat_g": 0.0,
    "carbs_g": 21.0,
    "sugar_g": 0.0,
    "sodium_mg": 70.0
  },
  "recommendation": "Recommended"
}
```

---

## Catatan Penting Pengembangan (Notes)
- **Modifikasi Bounding Box YOLO**: Terdapat logika khusus pada file `app/model/yolo_line.py` untuk menyesuaikan atau menahan perpanjangan batas lebar dari kotak pemotongan (crop) `bounding box` YOLO. Ini bertujuan agar kolom tabel persentase (% AKG) pada kemasan makanan tidak ikut dibaca dan mengacaukan teks nilai numerik OCR.
- **Optimasi Memori**: API ini secara otomatis akan memuat dan merendam tiga model berat sekaligus pada scope global saat proses _startup_. Sangat disarankan melakukan *deployment* pada *environment* (seperti VM, Cloud Run, atau EC2) yang menyediakan alokasi minimal memori RAM sebesar 1.5GB hingga 2GB untuk menjamin kelancaran *inference*.

---
*Developed for the DiaBites Application Ecosystem.*
