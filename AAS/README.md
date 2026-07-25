### OCR Plat Nomor Kendaraan menggunakan VLM (LMStudio + Python) ###

Project ini melakukan Optical Character Recognition (OCR) pada plat nomor kendaraan menggunakan Visual Language Model (VLM) yang dijalankan secara lokal lewat LMStudio, lalu diintegrasikan dengan Python. Dataset yang dipakai adalah Indonesian License Plate Dataset.

## Model yang Digunakan: ##
Model VLM: qwen2.5-vl-7b-instruct, dijalankan lewat LMStudio sebagai local server yang meniru format OpenAI API (/v1/chat/completions).

Cara Kerja Singkat
Baca file label (format YOLO: x_center y_center width height plate_text) untuk mendapatkan koordinat bounding box plat nomor beserta ground truth-nya.
Crop gambar sesuai bounding box, ditambah sedikit padding supaya teks plat tidak terpotong. Kalau hasil crop terlalu kecil, gambar akan di-resize dulu supaya lebih mudah dibaca model.
Gambar hasil crop dikirim ke LMStudio dalam format base64, dibarengi prompt yang meminta model hanya membalas nomor platnya saja.
Hasil prediksi dibersihkan (hanya huruf & angka, di-uppercase), lalu dibandingkan dengan ground truth menggunakan metrik Character Error Rate (CER).
Semua hasil disimpan ke results.csv dengan kolom: image, ground_truth, prediction, CER_score.
