import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import load_image, convert_to_grayscale, convert_to_hsv

def test_load_image():
    image_path = "tests/sample_image.jpg"
    image = load_image(image_path)
    assert image is not None, "Gambar tidak berhasil dimuat."
    assert image.shape[2] == 3, "Gambar harus dalam format BGR dengan 3 channel."

def test_convert_to_grayscale():
    image_path = "tests/sample_image.jpg"
    image = load_image(image_path)
    gray_image = convert_to_grayscale(image)
    assert gray_image is not None, "Konversi ke Grayscale gagal."
    assert len(gray_image.shape) == 2, "Grayscale image seharusnya memiliki 2 dimensi."

def test_convert_to_hsv():
    image_path = "tests/sample_image.jpg"
    image = load_image(image_path)
    hsv_image = convert_to_hsv(image)
    assert hsv_image is not None, "Konversi ke HSV gagal."
    assert hsv_image.shape[2] == 3, "HSV image harus memiliki 3 channel."