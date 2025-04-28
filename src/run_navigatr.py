import base64
import json
import os
import subprocess
import threading
import time
import cv2
import numpy as np
import torch
from src.webcam import capture_one_frame, connect_to_webcam, run_camera_service, view_processed_frames
from src.sharable_data import obj_res_queue, depth_res_queue, emot_res_queue

np.set_printoptions(threshold=1000)  # Default threshold value
torch.set_printoptions(profile="default")
script_dir = os.path.dirname(os.path.abspath(__file__))
config_paths_json = os.path.join(script_dir, "./config.json")


with open(config_paths_json, "r") as file:  
    navigatr_config = json.load(file)

if navigatr_config['dev_code'] != "expo":
    from src.obj_detect.run_model import run_obj_detect_model
    from src.distance.run_model import run_dep_detect_model

if navigatr_config['dev_code'] != "expo":
    camera_thread = threading.Thread(target=run_camera_service,
                                    args=(navigatr_config['camera_method'],
                                        navigatr_config['ip_cam_addr'],
                                        navigatr_config['test_toggle']), daemon=True)
    camera_thread.start()

    # Boot up the object detection model
    if navigatr_config['obj_detect_on']:
        obj_detect_thread = threading.Thread(target=run_obj_detect_model, args=(None, navigatr_config['test_toggle']), daemon=True)
        obj_detect_thread.start()  # Start inference in a separate thread

    if navigatr_config['depth_detect_on']:
        dep_detect_thread = threading.Thread(target=run_dep_detect_model, args=(None, navigatr_config['test_toggle']), daemon=True)
        dep_detect_thread.start()  # Start inference in a separate thread

    while camera_thread.is_alive(): # and obj_detect_thread.is_alive(): # should I include dep_detect_thread.is_alive()?
        time.sleep(10)

else:
    # Run sequentially
    ip_cam_addr = navigatr_config['ip_cam_addr']
    test_toggle = navigatr_config['test_toggle']

    #'''

    capture, raw_out, processed_out = connect_to_webcam(ip_cam_addr=ip_cam_addr, test_toggle=test_toggle)
    frame = capture_one_frame("webcam_single", capture, raw_out, test_toggle, navigatr_config['dev_code'])

    # Convert the frame to a bytes object
    frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
    print(f"SENDING: Sending frame with {len(frame_bytes)} bytes")

    success, encoded_img = cv2.imencode('.jpg', frame)
    if not success:
        print("ERROR: Failed to encode frame!")
        exit(1)

    input_image_path = "input.jpg" # Path to the input image that was captured
    output_json_path = "output.json" # Path to where output data will be stored

    with open(input_image_path, "wb") as f:
        f.write(encoded_img.tobytes())
    #'''
    display_thread = threading.Thread(target=view_processed_frames,
                                    args=([None, None, None], False), daemon=False)
    display_thread.start()

    # input_image_path = "test_input_img.jpg" # Path to the input image that was captured
    obj_output_json_path = "obj_output.json" # Path to where output data will be stored
    dep_output_json_path = "dep_output.json" # Path to where output data will be stored
    emot_output_json_path = "emot_output.json" # Path to where output data will be stored


    p1 = subprocess.Popen(
        ['conda', 'run', '-n', 'navi_env', 'python', '-m', 'src.run_obj', input_image_path, obj_output_json_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_data, stderr_data = p1.communicate()
    stderr_data = stderr_data.decode('utf-8') if stderr_data else ""

    if p1.returncode != 0:
        print(f"ERROR: run_obj failed:\n{stderr_data}")
        exit(1)

    with open(obj_output_json_path, "r") as f:
        result_packet = json.load(f)

    print("SUCCESS: Received data back from run_obj.py")

    # Decode image and display
    processed_img_bytes = base64.b64decode(result_packet["processed_frame"])
    processed_img_np = np.frombuffer(processed_img_bytes, dtype=np.uint8)
    processed_img = cv2.imdecode(processed_img_np, cv2.IMREAD_COLOR)

    result_packet["processed_frame"] = processed_img
    obj_results = result_packet["detections"]

    # push processed frame to queue
    obj_res_queue.append(result_packet)


    p2 = subprocess.Popen(
        ['conda', 'run', '-n', 'depth-pro', 'python', '-m', 'src.run_depth', input_image_path, dep_output_json_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_data, stderr_data = p2.communicate()
    stderr_data = stderr_data.decode('utf-8') if stderr_data else ""

    if p2.returncode != 0:
        print(f"ERROR: run_depth failed:\n{stderr_data}")
        exit(1)

    with open(dep_output_json_path, "r") as f:
        result_packet = json.load(f)
    
    depth_results = result_packet["processed_frame"]

    print("SUCCESS: Received data back from run_depth.py")

    # Decode image and display
    processed_img_bytes = base64.b64decode(result_packet["depth_image"])
    processed_img_np = np.frombuffer(processed_img_bytes, dtype=np.uint8)
    processed_img = cv2.imdecode(processed_img_np, cv2.IMREAD_COLOR)

    result_packet["processed_frame"] = processed_img

    # push processed frame to queue
    depth_res_queue.append(result_packet)


    # Find relevant objects and determine depth to objects
    confidence_threshold = navigatr_config['confidence_threshold']

    # processed_out = [None, None, None]
    # view_processed_frames(processed_out, False)
