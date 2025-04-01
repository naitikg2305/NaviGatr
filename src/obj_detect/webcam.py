
import os
import cv2
import json
from src.sharable_data import result_queue, frame_queue

script_dir = os.path.dirname(os.path.abspath(__file__))
config_paths_json = os.path.join(script_dir, "config_paths.json")

# Load JSON data from file
with open(config_paths_json, "r") as file:
    config_data = json.load(file)


def connect_to_webcam():
    cam_url = config_data['ip_cam_addr']
    cap = cv2.VideoCapture(cam_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not frame_queue.full():
            frame_queue.put(frame)

        if not result_queue.empty():
            result_packet = result_queue.get()

            # Access the processed frame for visualization
            processed_frame = result_packet["processed_frame"]
            detections = result_packet["detections"]
            timestamp = result_packet["timestamp"]

            print(f"Frame Timestamp: {timestamp}, Detections: {len(detections)} objects found.")

            cv2.imshow("Phone IP Camera", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    frame_queue.put(None)  # Stop the inference thread
    cv2.destroyAllWindows()

