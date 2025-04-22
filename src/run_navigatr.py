import json
import os
import sys
import threading
import time
from src.obj_detect import run_model
from src.webcam import run_camera_service

script_dir = os.path.dirname(os.path.abspath(__file__))
config_paths_json = os.path.join(script_dir, "./config.json")

with open(config_paths_json, "r") as file:  
    navigatr_config = json.load(file)

camera_thread = threading.Thread(target=run_camera_service,
                                 args=(navigatr_config['camera_method'],
                                       navigatr_config['ip_cam_addr'],
                                       navigatr_config['test_toggle']), daemon=True)
camera_thread.start()

# Boot up the object detection model
if navigatr_config['obj_detect_on']:
    obj_detect_thread = threading.Thread(target=run_model.run_obj_detect_model, args=(), daemon=True)
    obj_detect_thread.start()  # Start inference in a separate thread

while obj_detect_thread.is_alive() and camera_thread.is_alive():
    time.sleep(10)
