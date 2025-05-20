import subprocess
import numpy as np
import cv2
import time

# Test 1: Capture and Save an Image using libcamera-still
def capture_and_save_image():
    result = subprocess.run([
        "libcamera-still", "-n", "-t", "100", "-o", "test_output.jpg", "--encoding", "jpg"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("[ERROR] libcamera-still failed:", result.stderr.decode())
    else:
        print("[SUCCESS] Saved image as test_output.jpg")

# Test 2: Capture image to memory and display using OpenCV
def capture_and_show_cv2():
    result = subprocess.run([
        "libcamera-still", "-n", "-t", "100", "-o", "-", "--encoding", "jpg"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0 or len(result.stdout) == 0:
        print("[ERROR] Failed to capture image:", result.stderr.decode())
        return

    img_array = np.frombuffer(result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        print("[ERROR] Failed to decode image")
        return

    cv2.imshow("Captured Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Running Test 1: Save image to file")
    capture_and_save_image()
    time.sleep(1)

    print("\nRunning Test 2: Capture and show with OpenCV")
    capture_and_show_cv2()