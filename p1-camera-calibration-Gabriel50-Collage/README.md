[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Y_6WRaQb)
# Tugas Camera Calibration dengan OpenCV

## Deskripsi
Tugas ini bertujuan untuk memahami proses kalibrasi kamera menggunakan Python dan OpenCV.
Anda akan:
1. Deteksi sudut papan catur dari dataset.
2. Lakukan kalibrasi kamera (`cv2.calibrateCamera`).
3. Simpan parameter kamera ke file `.npz`.
4. Lakukan undistorsi pada 1 gambar contoh.
5. Tampilkan dan simpan hasil undistorsi.

## Petunjuk
- Lengkapi fungsi `load_chessboard_images`, `calibrate_camera`, `undistort_image`, dan `save_camera_parameters` di `src/camera_calib.py`.
- Simpan hasil di folder `outputs/`:
  - `camera_params.npz`
  - `undistorted_sample.jpg`
- Tuliskan laporan singkat di file `REPORT.md`:
  - Hasil undistorsi (sertakan screenshot)

## Penilaian
- Kelengkapan fungsi (40%)
- Kebenaran hasil kalibrasi & RMS < 1.0 px (40%)
- Kerapihan kode & laporan (20%)

## Menjalankan
```bash
pip install -r requirements.txt
python src/camera_calib.py
