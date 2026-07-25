## OCR Plat Nomor Kendaraan menggunakan VLM ##

Project ini melakukan Optical Character Recognition (OCR) pada plat nomor kendaraan menggunakan Visual Language Model (VLM) yang dijalankan secara lokal lewat LMStudio, lalu diintegrasikan dengan Python. Dataset yang dipakai adalah Indonesian License Plate Dataset.

## Model yang Digunakan ##
Model VLM: qwen2.5-vl-7b-instruct, dijalankan lewat LMStudio sebagai local server yang meniru format OpenAI API.

Cara Kerja Program:
1. Baca file label (format YOLO: x_center y_center width height plate_text) untuk mendapatkan koordinat bounding box plat nomor beserta ground truth-nya.
2. Crop gambar sesuai bounding box, ditambah sedikit padding supaya teks plat tidak terpotong. Kalau hasil crop terlalu kecil, gambar akan diresize supaya lebih mudah dibaca oleh model.
3. Gambar hasil crop dikirim ke LMStudio dalam format base64, dibarengi prompt yang meminta model hanya membalas nomor platnya saja.
4. Hasil prediksi dibersihkan, lalu dibandingkan dengan ground truth menggunakan metrik Character Error Rate (CER).
5. Semua hasil disimpan ke results.csv dengan kolom: image, ground_truth, prediction, CER_score.

## Persiapan Menjalankan Program ##
1. Install dependencies:
   pip install requests pillow

2. Jalankan LMStudio
- Buka aplikasi LMStudio.
- Download dan load model qwen2.5-vl-7b-instruct (atau model lainnya yang mendukung image input).
- Aktifkan Local Server di LMStudio

3. Unduh Dataset
   Untuk dataset yang dipakai, kita bisa download menggunakan link ini: 
   https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset
"Jika sudah di download, extract file dataset dan satukan dataset yang sudah di extract dengan file program pada satu folder yang sama"

5. Menjalankan Program
   - Sebelum menjalankan program, pastikan untuk mengaktifkan LMStudio dan model sudah ter-load lalu jalankan programnya.
   - Script akan memproses semua gambar pada folder test, mengirim tiap crop plat ke LMStudio, lalu menampilkan progress di terminal.
   - Setelah selesai, hasil evaluasi tersimpan di results.csv
