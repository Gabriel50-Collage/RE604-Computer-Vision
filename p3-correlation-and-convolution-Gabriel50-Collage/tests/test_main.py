import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import correlate, convolve

def test_correlate():
    image_path = "tests/sample_image.png"
    image = cv2.imread(image_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    output_student = correlate(image, kernel)
    output_expected = np.load("tests/output_correlate.npy")

    assert output_student is not None, "Gambar tidak berhasil dimuat."
    assert (output_student == output_expected).all(), "Hasil correlate tidak sesuai."

def test_convolve():
    image_path = "tests/sample_image.png"
    image = cv2.imread(image_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
    output_student = convolve(image, kernel)
    output_expected = np.load("tests/output_convolve.npy")
    assert output_student is not None, "Gambar tidak berhasil dimuat."
    assert (output_student == output_expected).all(), "Hasil convolve tidak sesuai."