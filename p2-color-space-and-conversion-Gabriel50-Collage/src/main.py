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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def convert_to_hsv(image):
    if len(image.shape) == 2:
        raise ValueError("Gambar grayscale tidak dapat dikonversi ke HSV secara langsung.")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv

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

        display_images({
            "Original": original,
            "Grayscale": grayscale,
            "HSV": hsv
        })

    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)