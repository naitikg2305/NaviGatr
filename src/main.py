import base64
import json
import os
import threading
import time
from io import BytesIO
import tqdm
import numpy as np
import cv2
# import torcph
from src.output_handler import get_output, text_list_to_opencv
# from src.text_to_speech import text_to_speech
from src.sharable_data import (thread_lock, output_stack_res, frame_queue,
                               obj_queue, depth_queue, emot_queue,
                               obj_res_queue, depth_res_queue, emot_res_queue,
                               colors, reset)
from src.webcam import capture_frame, connect_to_webcam, view_processed_frames
from src.obj_detect.run_model import run_obj_detect_model
from src.distance.run_model import run_dep_detect_model
from src.capture import capture_frame

import logging
logging.basicConfig(filename='app.log', level=logging.INFO)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

np.set_printoptions(threshold=1000)  # Default threshold value
# torch.set_printoptions(profile="default")
script_dir = os.path.dirname(os.path.abspath(__file__))
config_paths_json = os.path.join(script_dir, "./config.json")


with open(config_paths_json, "r") as file:  
    navigatr_config = json.load(file)

print(f"🧭🧭  \033[38;2;0;0;128mTHANK YOU FOR USING NAVIGATR. STARTING PROGRAM\033[0m  🧭🧭 ")
time.sleep(3)

# tts_lock = threading.Lock()
# def tts_thread_worker(text, test_toggle=navigatr_config['test_toggle']):
#     with tts_lock:
#         print(f"Thread6: TTS: Starting speech: {text}") if test_toggle else None
#         # text_to_speech(text)
#         print(f"Thread6: TTS: Done speech: {text}") if test_toggle else None

# Run sequentially
ip_cam_addr = navigatr_config['ip_cam_addr']
test_toggle = navigatr_config['test_toggle']

#capture, raw_out, processed_out = connect_to_webcam(ip_cam_addr=ip_cam_addr, test_toggle=test_toggle)


# display_thread = threading.Thread(target=view_processed_frames,
                                # args=([None, None, None], False), daemon=True)
# display_thread.start()
# view_processed_frames([None, None, None], False)

# tts_thread: threading.Thread = None
while True:
    
    with tqdm.tqdm(total=4, desc="Processing...", ncols=100) as pbar:
            image_bytes = capture_frame()

            # Convert bytes to NumPy array
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            print(image_np)
            # Decode the image (this returns a BGR OpenCV image)
            frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("cv2.imdecode failed — check if libcamera-still returned valid JPEG bytes.")

            if thread_lock.acquire():
                try:
                    frame_queue.put(frame)
                    obj_queue.put(frame)
                    depth_queue.put(frame)
                    emot_queue.put(frame)
                    print(f"ThreadM: Single frame added to queue. The frame_queue is now size: {len(frame_queue)}") if test_toggle else None
                except:
                    print("ThreadM: Frame queue is full. Skipping frame...")  if test_toggle else None
                thread_lock.release()
            print(f"ThreadM: Done capture_frame. Current time: {time.time()}") if test_toggle else None

            pbar.update(1)

            # Run objdet on frame in thread
            if navigatr_config['obj_detect_on']:
                obj_detect_thread = threading.Thread(target=run_obj_detect_model, args=(None, navigatr_config['test_toggle']), daemon=True)
                obj_detect_thread.start()  # Start inference in a separate thread

            pbar.update(1)

            # Run depthdet on frame in thread

            if navigatr_config['depth_detect_on']:
                dep_detect_thread = threading.Thread(target=run_dep_detect_model, args=(None, navigatr_config['test_toggle']), daemon=True)
                dep_detect_thread.start()  # Start inference in a separate thread

            pbar.update(1)

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
            
            pbar.update(1)
            print(get_output(True))


            # if tts_lock.locked():
            #     print("ThreadM: TTS in progress, skipping.") if test_toggle else None
            # else:
            #     print("ThreadM: TTS not in progress, starting output handler.") if test_toggle else None
            #     output_strings = get_output(test_toggle=test_toggle)

            #     if output_strings is not None:
            #         image = text_list_to_opencv(output_strings)
            #         thread_lock.acquire()
            #         output_stack_res.put(image)
            #         print(f"ThreadM: Placed image in output stack. Output stack size: {len(output_stack_res)}") if test_toggle else None
            #         thread_lock.release()

            #         # Start TTS thread with locking
            #         tts_thread = threading.Thread(target=tts_thread_worker, args=(output_strings[0],), daemon=True)
            #         tts_thread.start()

            pbar.update(1)
            #time.sleep(1)

    # print(f"ThreadM: Restarting Process")
    # if not display_thread.is_alive():
    #     break
