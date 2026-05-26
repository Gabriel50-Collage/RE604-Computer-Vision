import glob
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Configuration
# =========================
N_ROWS = 9
N_COLS = 6
CHESSBOARD_SIZE = (N_ROWS, N_COLS)

CRITERIA = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
CALIB_FILE = os.path.join(OUTPUT_DIR, "camera_params.npz")

MAX_SAMPLES = 15


# =========================
# Helper functions
# =========================
def generate_object_points():
    objp = np.zeros((N_ROWS * N_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:N_ROWS, 0:N_COLS].T.reshape(-1, 2)
    return objp


def save_camera_params(K, dist, rms):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(CALIB_FILE, K=K, dist=dist, rms=rms)


def calibrate_camera(objpoints, imgpoints, image_size):
    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise ValueError("No valid chessboard samples were collected.")

    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("RMS:", rms)
    print("K:\n", K)
    print("dist:\n", dist)

    save_camera_params(K, dist, rms)
    return K, dist


def get_undistorted_frame(frame, K, dist):
    h, w = frame.shape[:2]
    newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undistorted = cv.undistort(frame, K, dist, None, newK)
    return undistorted


# =========================
# Calibration from images
# =========================
def calibrate_from_images(show_pics=True):
    image_paths = glob.glob(os.path.join(DATA_DIR, "*.jpg"))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {DATA_DIR}")

    objpoints = []
    imgpoints = []
    objp = generate_object_points()
    gray = None

    for image_path in image_paths:
        img = cv.imread(image_path)
        if img is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if found:
            corners_refined = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), CRITERIA
            )
            objpoints.append(objp)
            imgpoints.append(corners_refined)

            if show_pics:
                preview = img.copy()
                cv.drawChessboardCorners(preview, CHESSBOARD_SIZE, corners_refined, found)
                cv.imshow("Chessboard", preview)
                cv.waitKey(200)

    cv.destroyAllWindows()

    if gray is None:
        raise ValueError("Could not read any valid images.")

    image_size = gray.shape[::-1]
    return calibrate_camera(objpoints, imgpoints, image_size)


# =========================
# Calibration from webcam
# =========================
def calibrate_from_webcam():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    objpoints = []
    imgpoints = []
    objp = generate_object_points()

    count = 0
    gray = None

    print("Press 's' to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        display = frame.copy()

        if found:
            cv.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)

        cv.putText(
            display,
            f"{count}/{MAX_SAMPLES}",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv.imshow("Calibration", display)
        key = cv.waitKey(1) & 0xFF

        if key == ord("s") and found:
            corners_refined = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), CRITERIA
            )
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            count += 1
            print(f"Captured sample {count}/{MAX_SAMPLES}")

        elif key == ord("q"):
            break

        if count >= MAX_SAMPLES:
            break

    cap.release()
    cv.destroyAllWindows()

    if gray is None:
        raise ValueError("No frames were captured from webcam.")

    image_size = gray.shape[::-1]
    return calibrate_camera(objpoints, imgpoints, image_size)


# =========================
# Save sample images
# =========================
def save_sample(K, dist):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not capture image from webcam.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    original_path = os.path.join(OUTPUT_DIR, "original.jpg")
    undistorted_path = os.path.join(OUTPUT_DIR, "undistorted.jpg")

    cv.imwrite(original_path, frame)

    undistorted = get_undistorted_frame(frame, K, dist)
    cv.imwrite(undistorted_path, undistorted)

    print("Saved images:")
    print(f"- {original_path}")
    print(f"- {undistorted_path}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(undistorted, cv.COLOR_BGR2RGB))
    plt.title("Undistorted")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# Live distortion removal
# =========================
def remove_distortion(K, dist):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        undistorted = get_undistorted_frame(frame, K, dist)

        cv.imshow("Original", frame)
        cv.imshow("Undistorted", undistorted)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


# =========================
# Main
# =========================
if __name__ == "__main__":
    mode = "webcam"  # change to "dataset" if using images in data/raw

    if mode == "dataset":
        K, dist = calibrate_from_images(show_pics=True)
    else:
        K, dist = calibrate_from_webcam()

    save_sample(K, dist)
    remove_distortion(K, dist)