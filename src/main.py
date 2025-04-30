import base64
import json
import os
import threading
import time
import cv2
import numpy as np
import torch
from src.webcam import capture_frame, connect_to_webcam, view_processed_frames
from src.obj_detect.run_model import run_obj_detect_model
from src.distance.run_model import run_dep_detect_model

np.set_printoptions(threshold=1000)  # Default threshold value
torch.set_printoptions(profile="default")
script_dir = os.path.dirname(os.path.abspath(__file__))
config_paths_json = os.path.join(script_dir, "./config.json")


with open(config_paths_json, "r") as file:  
    navigatr_config = json.load(file)


# Run sequentially
ip_cam_addr = navigatr_config['ip_cam_addr']
test_toggle = navigatr_config['test_toggle']



capture, raw_out, processed_out = connect_to_webcam(ip_cam_addr=ip_cam_addr, test_toggle=test_toggle)


display_thread = threading.Thread(target=view_processed_frames,
                                args=([None, None, None], False), daemon=True)
display_thread.start()

while True:
    print("")
    # Capture frame
    frame = capture_frame(navigatr_config['camera_method'], capture, raw_out, test_toggle)

    print(f"ThreadM: Done capture_frame. Current time: {time.time()}")

    # Run objdet on frame in thread
    if navigatr_config['obj_detect_on']:
        obj_detect_thread = threading.Thread(target=run_obj_detect_model, args=(None, navigatr_config['test_toggle']), daemon=True)
        obj_detect_thread.start()  # Start inference in a separate thread


    # Run depthdet on frame in thread

    if navigatr_config['depth_detect_on']:
        dep_detect_thread = threading.Thread(target=run_dep_detect_model, args=(None, navigatr_config['test_toggle']), daemon=True)
        dep_detect_thread.start()  # Start inference in a separate thread


    # Wait for both threads to finish
    try:
        obj_detect_thread.join()
    except: # Fails if object detection model is turned off (i.e. thread never started)
        pass
    try:
        dep_detect_thread.join()
    except: # Fails if depth detection model is turned off (i.e. thread never started)
        pass

    print(f"ThreadM: Done running models. Current time: {time.time()}")
    ## OUPUT HANDLER BELOW

    if not display_thread.is_alive():
        break
