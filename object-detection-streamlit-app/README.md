# 🔍 Object Detection Streamlit Application

Aplikasi web untuk deteksi objek menggunakan Streamlit dan YOLOv8. Aplikasi ini mendukung deteksi objek pada gambar, URL, video, dan webcam real-time.

## ✨ Fitur Utama

- 📷 **Deteksi Gambar**: Upload dan deteksi objek pada gambar
- 🔗 **Deteksi URL**: Deteksi objek langsung dari URL gambar di internet
- 🎥 **Deteksi Video**: Proses video dan deteksi objek frame by frame
- 📹 **Webcam Real-time**: Deteksi objek secara real-time (dalam development)
- 🎯 **Multiple Models**: Dukungan YOLOv8 dan model custom
- ⚙️ **Konfigurasi Fleksibel**: Atur confidence threshold dan NMS
- 📊 **Statistik Deteksi**: Tampilan statistik hasil deteksi
- 💾 **Download Results**: Download hasil deteksi

## 🚀 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/wondrv/object-detection-streamlit-app.git
cd object-detection-streamlit-app
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```

## 📖 Cara Penggunaan

### Deteksi Gambar
1. Pilih tab "Image Detection"
2. Upload gambar (JPG, PNG, BMP)
3. Atur confidence threshold di sidebar
4. Klik "Detect Objects"
5. Lihat hasil deteksi dan statistik

### Deteksi URL
1. Pilih tab "URL Detection"
2. Masukkan URL gambar langsung dari internet
3. Klik "Detect from URL"
4. Lihat hasil deteksi dan download jika diperlukan
5. Coba contoh URL yang disediakan

### Deteksi Video
1. Pilih tab "Video Detection"
2. Upload video (MP4, AVI, MOV, MKV)
3. Klik "Process Video"
4. Download hasil video yang sudah diproses

### Konfigurasi Model
- **YOLOv8n**: Model tercepat, akurasi standard
- **YOLOv8s**: Keseimbangan speed dan akurasi
- **YOLOv8m**: Akurasi tinggi, speed sedang
- **Custom Model**: Gunakan model yang sudah ditraining

## 🎯 Custom Model

Untuk menggunakan model custom:

1. Letakkan file model (.pt) di folder `models/`
2. Update path di `config/config.yaml`
3. Pilih "Custom Model" di aplikasi

## 📁 Struktur Proyek

```
object-detection-streamlit-app/
├── app.py                 # Aplikasi utama Streamlit
├── requirements.txt       # Dependencies Python
├── README.md             # Dokumentasi
├── config/
│   └── config.yaml       # Konfigurasi aplikasi
├── models/               # Model files
├── utils/                # Utility functions
├── assets/               # Sample files
└── docs/                 # Dokumentasi lengkap
```

## 🔧 Konfigurasi

Edit file `config/config.yaml` untuk mengubah:
- Model default
- Threshold confidence dan NMS
- Ukuran maksimum file
- Pengaturan UI

## 📚 Dokumentasi

- [Installation Guide](docs/installation.md)
- [Usage Instructions](docs/usage.md)
- [API Reference](docs/api_reference.md)

## 🎬 Demo Video

[Link ke video demo YouTube akan ditambahkan di sini]

## 🤝 Kontribusi

1. Fork repository
2. Buat branch fitur (`git checkout -b feature/amazing-feature`)
3. Commit perubahan (`git commit -m 'Add amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Buat Pull Request

## 📄 Lisensi

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Kontak

wondrv - niiellpz@gmail.com

Project Link: [https://github.com/wondrv/object-detection-streamlit-app](https://github.com/wondrv/object-detection-streamlit-app)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)