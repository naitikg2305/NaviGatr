import threading
import os
import time
import torch
from nanodet.util import cfg, load_config, Logger
from demo.demo import Predictor
from nanodet.util import overlay_bbox_cv
import json
from webcam import connect_to_webcam

script_dir = os.path.dirname(os.path.abspath(__file__))
config_paths_json = os.path.join(script_dir, "config_paths.json")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cpu')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

## Path to model files
obj_detect_dir = os.path.dirname(os.path.abspath(__file__))  # This script's directory
src_dir = os.path.dirname(obj_detect_dir)
config_paths_json = os.path.join(obj_detect_dir, "config_paths.json")
with open(config_paths_json, "r") as file:
    config_data = json.load(file)
print()
config_path = os.path.join(src_dir, config_data['config_file'])
model_path = os.path.join(src_dir, config_data['model_file'])

load_config(cfg, config_path)
logger = Logger(-1, use_tensorboard=False)

predictor = Predictor(cfg, model_path, logger, device=device)

''' # Sequential handling of frames and processing
cap = cv2.VideoCapture(video_path)

# Video details
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()

    # Resizing video frame to model's expected input size
    input_img = cv2.resize(frame, (416, 416))

    meta, res = predictor.inference(input_img)

    result = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.35)

    cv2.imshow("Object Detection Vid Frame", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
''' # Parallel handling of frames and processing

from sharable_data import frame_queue, result_queue

def inference_thread():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break  # Stop if termination signal received

        start_time = time.time()  # Timestamp
        meta, res = predictor.inference(frame)

        processed_frame = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.35)

        # Data being bundled as a packet
        result_packet = {
            "timestamp": start_time,
            "detections": res[0],  # Raw detection data
            "processed_frame": processed_frame
        }

        result_queue.put(result_packet)
        frame_queue.task_done()


threading.Thread(target=inference_thread, daemon=True).start()  # Start inference in a separate thread

connect_to_webcam(frame_queue, result_queue)

# This line will not run until webcam is closed.

#'''
