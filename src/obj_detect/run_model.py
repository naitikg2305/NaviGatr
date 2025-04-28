import os
import time
import torch

'''Remove after replacement of nanodet''' 
# from nanodet.util import cfg, load_config, Logger
# from demo.demo import Predictor
# from nanodet.util import overlay_bbox_cv

'''Replacement code:'''
from ultralytics import YOLO

import json
from src.sharable_data import thread_lock, obj_queue, obj_res_queue

# Requires Windows OS, remove if using Linux
import ctypes
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00008000) # Makes this thread highest priority

script_dir = os.path.dirname(os.path.abspath(__file__))
config_paths_json = os.path.join(script_dir, "config.json")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cpu') # Change to 'cuda' to run on GPU

"""
"The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives
for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as
forward and backward convolution, attention, matmul, pooling, and normalization." [Quoted]
https://developer.nvidia.com/cudnn
"""
torch.backends.cudnn.enabled = True  # (https://pytorch.org/docs/stable/backends.html)
torch.backends.cudnn.benchmark = True  # (https://pytorch.org/docs/stable/backends.html)

## Path to model files
obj_detect_dir = os.path.dirname(os.path.abspath(__file__))  # This script's directory
config_paths_json = os.path.join(obj_detect_dir, "config.json")
with open(config_paths_json, "r") as file:
    config_data = json.load(file)
config_path = os.path.join(obj_detect_dir, config_data['config_file'])
model_path = os.path.join(obj_detect_dir, config_data['model_file'])

'''Remove after replacement of nanodet'''
# load_config(cfg, config_path)
# logger = Logger(-1, use_tensorboard=False) # Disable Tensorboard (an ML tool for visualizing log data)
# predictor = Predictor(cfg, model_path, logger, device=device)

'''Replacement code:'''
## Load the YOLO11 model
# model = YOLO(model_path)
## Export the model to ONNX format
# model.export(format="onnx")  # creates 'yolo11n.onnx'
# Load the exported ONNX model
onnx_model = YOLO("yolo11n.onnx")



def run_obj_detect_model(frame=None, test_toggle: bool = False):
    """
    Either runs on a single frame (given as an argument),
    or runs on frames from a continously populated frame queue.
    Defaults to expecting a frame queue.

    Assumes camera frames are being generated faster than model can handle them

    Args:
        frame (numpy.ndarray): Frame to run inference on

    Returns:
        None

    """
    print(f"Booting up object detection model...")
    # Give time for camera service to generate its first frame
    time.sleep(15)
    print(f"Running object detection model...")

    frame_given = True  # Assume frame was given
    while True:
        if thread_lock.acquire():
            if frame is None: # If frame was not given, get from queue and override frame_given assumption
                frame_given = False
                print(f"Thread3: Getting frame from obj_queue of queue size: {len(obj_queue)}")
                try:
                    frame = obj_queue.get()
                except: # If its still None, no more frames in queue
                    print(f"Thread3: Could not get frame from obj_queue")
                    break # Stop if no more frames in queue
            print(f"Thread3: Dimensions of input frame: {frame.shape}")

            '''Remove after replacement of nanodet'''
            # meta, res = predictor.inference(frame)  # Run inference on a frame and get results
            # Run inference            
            # results = onnx_model("https://ultralytics.com/images/bus.jpg")
            start_watch = time.time()
            results = onnx_model.predict(frame)
            stop_watch = time.time()
            elapsed_time = stop_watch - start_watch
            results = results[0]

            thread_lock.release()
        
            if test_toggle: # Log inference outputs
                log_inference_outputs(results)

            '''Remove after replacement of nanodet'''
            # processed_frame = overlay_bbox_cv(meta['raw_img'][0], res[0],
            #                                 cfg.class_names, score_thresh=0.35) # Draw bounding boxes
            # print(f"\nThread3: Dimensions of processed_frame: {processed_frame.shape}")
            # # Data being bundled as a packet
            # result_packet = {
            #     "timestamp": start_time,
            #     "detections": res[0],  # Raw detection data
            #     "processed_frame": processed_frame
            # }

            frame = None # Clear frame
            # Data being bundled as a packet
            result_packet = {
                "inference_time": elapsed_time,
                "detections": [json.loads(results.to_json())],  # all detections
                "processed_frame": results.plot(),           # list of annotated images
                "verbose": [r.verbose() for r in results]                  # human-readable summary for each image
            }


            print(f"Thread3: Detected objects in {elapsed_time} seconds")
            obj_res_queue.append(result_packet)
            print(f"Thread3: obj_res_queue size: {len(obj_res_queue)}")

            # If frame was passed as an argument, break out of the loop (i.e. only run once)
            if frame_given:
                return result_packet

def log_inference_outputs(res):

    '''Remove after replacement of nanodet'''
    # # print(f"\nThread3: meta['img_info'] shape: {meta['img_info'].shape}, meta['img_info']: {meta['img_info']}")
    # print(f"\nThread3: meta['img_info']: {meta['img_info']}")
    # # print(f"\nThread3: meta['raw_img'] shape: {meta['raw_img'].shape}, meta['raw_img']: {meta['raw_img']}")
    # print(f"\nThread3: meta['raw_img']: {meta['raw_img']}")
    # # print(f"\nThread3: meta['img'] Tensor shape: {meta['img'].shape}, meta['img']: {meta['img']}")
    # print(f"\nThread3: meta['img']: {meta['img']}")
    # print(f"\nThread3: meta['warp_matrix']: {meta['warp_matrix']}")
    # # loop through all 80 classification labels
    # for i in range(80):
    #     # print(f"\nThread3: res[0][{i}] shape: {res[0][i].shape}, Classification {i} details: {res[0][i]}")
    #     print(f"\nThread3: Classification {i} details: {res[0][i]}")
    print(f"\nThread3: results: {res}")
    return

