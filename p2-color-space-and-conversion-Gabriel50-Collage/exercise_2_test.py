import cv2
import os
import sys

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File tidak ditemukan di lokasi: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gambar gagal dimuat, periksa format file: {image_path}")
    return img

def convert_to_grayscale(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_hsv(image):
    if len(image.shape) == 2:
        raise ValueError("Gambar grayscale tidak bisa dikonversi ke HSV.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def _to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def _to_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def gray_at_pixel(image, x, y):
    gray = convert_to_grayscale(image)
    return int(gray[y, x])

def hue_at_pixel(image, x, y):
    hsv = _to_hsv(image)
    return int(hsv[y, x, 0])

def saturation_at_pixel(image, x, y):
    hsv = _to_hsv(image)
    return int(hsv[y, x, 1])

def brightness_at_pixel(image, x, y):
    hsv = _to_hsv(image)
    return int(hsv[y, x, 2])

def lightness_at_pixel(image, x, y):
    hls = _to_hls(image)
    return int(hls[y, x, 1])

def display_images(windows: dict):
    for title, img in windows.items():
        cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\ASUS TUF GAMING\Documents\Semester 6\RE604 Computer Vision\Color Space & Camera Conversion\p2-color-space-and-conversion-Gabriel50-Collage\tests\sample_image.jpg"

    try:
        original = load_image(image_path)
        grayscale = convert_to_grayscale(original)
        hsv = convert_to_hsv(original)

        test_x, test_y = 50, 50
        print(f"Gray   at ({test_x},{test_y}): {gray_at_pixel(original, test_x, test_y)}")
        print(f"Hue    at ({test_x},{test_y}): {hue_at_pixel(original, test_x, test_y)}")
        print(f"Sat    at ({test_x},{test_y}): {saturation_at_pixel(original, test_x, test_y)}")
        print(f"Value  at ({test_x},{test_y}): {brightness_at_pixel(original, test_x, test_y)}")
        print(f"Light  at ({test_x},{test_y}): {lightness_at_pixel(original, test_x, test_y)}")

        display_images({
            "Original": original,
            "Grayscale": grayscale,
            "HSV": hsv
        })

    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)