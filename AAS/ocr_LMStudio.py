import base64
import csv
import io
import os
import re
import sys
import time
import requests
from PIL import Image

# ──────────────────────────── KONFIGURASI ────────────────────────────
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "qwen2.5-vl-7b-instruct"

IMAGES_DIR = "images/test"
LABELS_DIR = "labelswithLP/test"
OUTPUT_CSV = "results.csv"
PROMPT = (
    "Read the license plate number in this cropped image. "
    "Return ONLY the alphanumeric characters (letters and digits) without any spaces, "
    "extra text, or explanation."
)
CROP_PADDING = 0.15        # margin tambahan agar plat tidak terpotong terlalu rapat
MIN_CROP_WIDTH = 200       # di bawah ini crop akan di-resize ke RESIZE_WIDTH
RESIZE_WIDTH = 400         # lebar crop setelah resize (tinggi mengikuti proporsi)
MAX_RETRIES = 2            # jumlah coba ulang jika request gagal
RETRY_DELAY = 2            # detik jeda antar retry
# ─────────────────────────────────────────────────────────────────────


def parse_label_file(path):
    """
    Kembalikan list of (x_center, y_center, width, height, plate_text)
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            x_center, y_center, width, height = (float(v) for v in parts[1:5])
            plate_text = parts[5]
            entries.append((x_center, y_center, width, height, plate_text))
    return entries


def crop_plate(image, x_center, y_center, width, height):
    """Potong area plat dengan padding, lalu resize jika terlalu kecil."""
    img_w, img_h = image.size
    box_w = width * img_w
    box_h = height * img_h
    cx = x_center * img_w
    cy = y_center * img_h

    pad_w = box_w * CROP_PADDING
    pad_h = box_h * CROP_PADDING

    left   = max(0, cx - box_w/2 - pad_w)
    top    = max(0, cy - box_h/2 - pad_h)
    right  = min(img_w, cx + box_w/2 + pad_w)
    bottom = min(img_h, cy + box_h/2 + pad_h)

    crop = image.crop((left, top, right, bottom))

    # Jika crop sangat kecil, perbesar agar teks terbaca oleh model
    if crop.width < MIN_CROP_WIDTH:
        ratio = RESIZE_WIDTH / crop.width
        new_height = int(crop.height * ratio)
        crop = crop.resize((RESIZE_WIDTH, new_height), Image.LANCZOS)

    return crop


def encode_image_base64(image):
    buffer = io.BytesIO()
    # Simpan dengan kualitas tinggi agar teks tetap tajam
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def clean_plate_text(text):
    """Bersihkan teks: hanya huruf & angka, uppercase."""
    text = text.strip().upper()
    # Hapus semua karakter selain A-Z, 0-9
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def query_lmstudio(cropped_image):
    """Kirim gambar crop ke LM Studio, dengan retry."""
    b64 = encode_image_base64(cropped_image)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 20,   # plat nomor biasanya pendek
    }

    last_exception = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = requests.post(LMSTUDIO_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            if "choices" not in data or not data["choices"]:
                raise RuntimeError("Respons tidak memiliki 'choices'")
            content = data["choices"][0]["message"]["content"]
            # Ambil baris pertama jika multi-baris
            first_line = content.split("\n")[0].strip()
            return first_line if first_line else content.strip()
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise last_exception


def levenshtein_distance(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def compute_cer(gt, pred):
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein_distance(gt, pred) / len(gt)


def main():
    if not os.path.isdir(IMAGES_DIR) or not os.path.isdir(LABELS_DIR):
        print(f"Folder tidak ditemukan: {IMAGES_DIR} dan/atau {LABELS_DIR}")
        sys.exit(1)

    label_files = sorted(
        f for f in os.listdir(LABELS_DIR) if f.lower().endswith(".txt")
    )
    rows = []
    total_cer = 0.0
    count = 0

    for label_file in label_files:
        stem = os.path.splitext(label_file)[0]
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(IMAGES_DIR, stem + ext)
            if os.path.isfile(candidate):
                image_path = candidate
                break
        if image_path is None:
            print(f"Gambar untuk {label_file} tidak ditemukan, lewati.")
            continue

        entries = parse_label_file(os.path.join(LABELS_DIR, label_file))
        if not entries:
            continue

        image = Image.open(image_path).convert("RGB")

        for idx, (x, y, w, h, gt_text) in enumerate(entries, 1):
            ground_truth = clean_plate_text(gt_text)

            # Abaikan jika bounding box tidak valid
            if w <= 0 or h <= 0:
                print(f"[WARNING] {stem} plat {idx}: bounding box tidak valid, lewati.")
                continue

            cropped = crop_plate(image, x, y, w, h)
            try:
                raw_prediction = query_lmstudio(cropped)
            except Exception as e:
                print(f"[ERROR] {stem} plat {idx}: {e}")
                raw_prediction = ""

            prediction = clean_plate_text(raw_prediction)
            cer = compute_cer(ground_truth, prediction)
            total_cer += cer
            count += 1

            row_img_name = f"{stem}_{idx}{os.path.splitext(image_path)[1]}"
            rows.append({
                "image": row_img_name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "CER_score": round(cer, 4),
            })
            print(f"{row_img_name} | GT: {ground_truth} | Pred: {prediction} | CER: {cer:.4f}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "prediction", "CER_score"])
        writer.writeheader()
        writer.writerows(rows)

    if count:
        print(f"\nJumlah data: {count}")
        print(f"Rata-rata CER: {total_cer/count:.4f}")
        print(f"Hasil disimpan di {OUTPUT_CSV}")


if __name__ == "__main__":
    main()