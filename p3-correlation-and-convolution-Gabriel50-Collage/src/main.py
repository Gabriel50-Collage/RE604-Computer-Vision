import numpy as np
import cv2
from scipy.signal import correlate2d, convolve2d

def map_to_8bit(img_in):
    img_out = None
    img_out = ((img_in - img_in.min()) * 255/(img_in.max() - img_in.min())).astype(np.uint8)
    return img_out

def correlate(img_in, kernel):
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY).astype(np.float64)
    result = correlate2d(gray, kernel, mode='full', boundary='fill', fillvalue=0)
    return result

def convolve(img_in, kernel):
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY).astype(np.float64)
    result = convolve2d(gray, kernel, mode='same', boundary='fill', fillvalue=0)
    return result

def main():
    image_path = r"C:\Users\ASUS TUF GAMING\Documents\Semester 6\RE604 Computer Vision\p3-correlation-and-convolution-Gabriel50-Collage\tests\sample_image.png"
    bgr_image = cv2.imread(image_path)

    kernel_1 = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    coorelation_output = correlate(bgr_image, kernel_1)

    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    output_x = convolve(bgr_image, sobel_x)
    output_y = convolve(bgr_image, sobel_y)
    sobel_output = np.sqrt(output_x**2 + output_y**2)

    cv2.imshow("Correlation Output", map_to_8bit(coorelation_output))
    cv2.imshow("Sobel X", map_to_8bit(output_x))
    cv2.imshow("Sobel Y", map_to_8bit(output_y))
    cv2.imshow("Sobel Output", map_to_8bit(sobel_output))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()