import base64
import json
import os
import threading
import time
from tqdm.rich import tqdm
import cv2
import numpy as np
import torch
from src.output_handler import get_output, text_list_to_opencv
# from src.text_to_speech import text_to_speech
from src.sharable_data import thread_lock, output_stack_res
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
    for task in tqdm(range(4), desc="Processing...", ncols=100):
        print(f"Inside progress bar with task: {task}")
        if task == 0:
            # Capture frame
            frame = capture_frame(navigatr_config['camera_method'], capture, raw_out, test_toggle)

            print(f"ThreadM: Done capture_frame. Current time: {time.time()}")

            continue

        if task == 1:
            # Run objdet on frame in thread
            if navigatr_config['obj_detect_on']:
                obj_detect_thread = threading.Thread(target=run_obj_detect_model, args=(None, navigatr_config['test_toggle']), daemon=True)
                obj_detect_thread.start()  # Start inference in a separate thread


            # Run depthdet on frame in thread

            if navigatr_config['depth_detect_on']:
                dep_detect_thread = threading.Thread(target=run_dep_detect_model, args=(None, navigatr_config['test_toggle']), daemon=True)
                dep_detect_thread.start()  # Start inference in a separate thread

            continue


        if task == 2:
            # Wait for both threads to finish
            try:
                obj_detect_thread.join()
            except: # Fails if object detection model is turned off (i.e. thread never started)
                pass
            try:
                dep_detect_thread.join()
            except: # Fails if depth detection model is turned off (i.e. thread never started)
                pass

            continue
        
        if task == 3:

            print(f"ThreadM: Done running models. Current time: {time.time()}")
            ## OUPUT HANDLER BELOW
            output_strings = get_output(True)

            if output_strings is not None:
                image = text_list_to_opencv(output_strings)
                thread_lock.acquire()
                output_stack_res.put(image)
                print(f"ThreadM: Placed image in output stack. Output stack size: {len(output_stack_res)}")
                thread_lock.release()
                # for string in output_strings:
                #     text_to_speech(string)

            continue
    print(f"ThreadM: Restarting Process")
    if not display_thread.is_alive():
        break
