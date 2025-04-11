import os
import cv2
import json
import numpy as np
from queue import Queue
import time
# from picamera2 import Picamera2 # Only available on Linux-based systems
from src.sharable_data import result_queue, frame_queue

script_dir = os.path.dirname(os.path.abspath(__file__))
config_paths_json = os.path.join(script_dir, "./obj_detect/config.json")

# Load JSON data from file
with open(config_paths_json, "r") as file:
    config_data = json.load(file)

''' Uncomment on Linux-based system
def connect_to_rpi_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
    picam2.start()

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR (OpenCV format)

        if not frame_queue.full():
            frame_queue.put(frame)

        if not result_queue.empty():
            result_packet = result_queue.get()

            # Access the processed frame for visualization
            processed_frame = result_packet["processed_frame"]
            detections = result_packet["detections"]
            timestamp = result_packet["timestamp"]

            print(f"Frame Timestamp: {timestamp}, Detections: {len(detections)} objects found.")

            cv2.imshow("Raspberry Pi Camera", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    picam2.stop()
    frame_queue.put(None)  # Stop the inference thread
    cv2.destroyAllWindows()


def test_picam():
    time_now = time.ctime()

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
    picam2.start()

    frame = picam2.capture_array()
    cv2.imwrite(f"picam_img_{time_now}.jpg", frame)
    picam2.stop()

    print(f"picam test image as picam_img_{time_now}.jpg")
#'''
'''
def connect_to_webcam(test_toggle: bool = False):
    cam_url = config_data['ip_cam_addr']
    cap = cv2.VideoCapture(cam_url)

    if test_toggle:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps) if fps and fps > 0 else 30  # fallback if FPS not detected #
        fps = 15

        fourcc = cv2.VideoWriter_fourcc(*'MJPG') #
        raw_out = cv2.VideoWriter("raw_feed.avi", fourcc, fps, (frame_width, frame_height)) #
        processed_out = cv2.VideoWriter("processed_feed.avi", fourcc, fps, (frame_width, frame_height)) #


    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        print("Frame shape:", frame.shape)
        print(f"\nFrame width: {frame_width}, Frame height: {frame_height}\n")
        if not ret:
            break

        if test_toggle:
            raw_out.write(frame)

        try:
            frame_queue.put_nowait(frame)
        except:
            print("Frame queue is full. Skipping frame...")

        try:
            result_packet = result_queue.get(timeout=1.0)  # avoid blocking forever
        except:
            continue

        # Access the processed frame for visualization
        processed_frame = result_packet["processed_frame"]
        detections = result_packet["detections"]
        timestamp = result_packet["timestamp"]

        

        # Inside loop:
        if processed_frame is not None:
            if test_toggle:
                processed_out.write(processed_frame.copy())  # <-- KEY FIX
                frame_count += 1
                print(f"[Saved Frame #{frame_count}] Timestamp: {timestamp}, Detections: {len(detections)}")

        cv2.imshow("Phone IP Camera", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    if test_toggle:
        raw_out.release()
        processed_out.release()
    frame_queue.put(None)  # Stop the inference thread
    cv2.destroyAllWindows()
'''
def connect_to_webcam(test_toggle: bool = False):
    print(f"test_toggle is: {test_toggle}\n\n")
    cam_url = config_data['ip_cam_addr']
    cap = cv2.VideoCapture(cam_url)

    if test_toggle:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps) if fps and fps > 0 else 30  # fallback if FPS not detected
        fps = 15  # Set a reasonable FPS manually (make sure this matches your camera feed FPS)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        raw_out = cv2.VideoWriter("raw_feed.avi", fourcc, fps, (frame_width, frame_height))
        processed_out = cv2.VideoWriter("processed_feed.avi", fourcc, fps, (frame_width, frame_height))

    frame_count = 0  # Keep track of frame count globally

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Frame shape: {frame.shape}")
        print(f"Frame width: {frame_width}, Frame height: {frame_height}")
        
        if frame is None:
            print("\nFrame is None\n")

        if test_toggle:
            raw_out.write(frame)
            print(f"\nFRAME WAS WRITTEN\n")

        try:
            frame_queue.put_nowait(frame)
        except:
            print("Frame queue is full. Skipping frame...")

        try:
            result_packet = result_queue.get(timeout=1.0)  # avoid blocking forever
        except:
            continue

        # Access the processed frame for visualization
        processed_frame = result_packet["processed_frame"]
        detections = result_packet["detections"]
        timestamp = result_packet["timestamp"]

        if processed_frame is not None:
            print(f"Processed frame is not none")
            if test_toggle:
                processed_out.write(processed_frame.copy())  # Write processed frame to file
                frame_count += 1  # Increment frame count
                print(f"[Saved Frame #{frame_count}] Timestamp: {timestamp}, Detections: {len(detections)}")

        cv2.imshow("Phone IP Camera", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    if test_toggle:
        raw_out.release()
        processed_out.release()
        print(f"Saved videos were released")
    frame_queue.put(None)  # Stop the inference thread
    cv2.destroyAllWindows()




