import base64
import os
import sys
import time
import cv2
import torch

'''Remove after replacement of nanodet''' 
# from nanodet.util import cfg, load_config, Logger
# from demo.demo import Predictor
# from nanodet.util import overlay_bbox_cv

'''Replacement code:'''
from ultralytics import YOLO

import json
from src.sharable_data import thread_lock, obj_queue, obj_res_queue, output_stack, colors, reset

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

'''
Potential Error:

  File "C:/Users/<user>/anaconda3/lib/site-packages/ultralytics/utils/checks.py", line 541, in check_file
    raise FileNotFoundError(f"'{file}' does not exist")
FileNotFoundError: 'yolo11n.onnx' does not exist


Temp solution:

Go place 'yolo11n.onnx' in C:/Users/<user>/anaconda3/envs/<env>/Lib/site-packages 
or in "C:/Users/<user>/anaconda3/Lib/site-packages"

'''

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
    print(f"{colors['olive_green']}Thread3: Running object detection model... {reset}")

    frame_given = True  # Assume frame was given
    while True:
        if thread_lock.acquire():
            if frame is None: # If frame was not given, get from queue and override frame_given assumption
                frame_given = False
                print(f"{colors['olive_green']}Thread3: Getting frame from obj_queue of queue size: {len(obj_queue)} {reset}") if test_toggle else None
                try:
                    frame = obj_queue.get()
                except: # If its still None, no more frames in queue
                    print(f"{colors['olive_green']}Thread3: Could not get frame from obj_queue {reset}") if test_toggle else None
                    thread_lock.release()
                    break # Stop if no more frames in queue
            thread_lock.release()
            print(f"{colors['olive_green']}Thread3: Dimensions of input frame: {frame.shape} {reset}") if test_toggle else None

            '''Remove after replacement of nanodet'''
            # meta, res = predictor.inference(frame)  # Run inference on a frame and get results
            # Run inference            
            # results = onnx_model("https://ultralytics.com/images/bus.jpg")
            start_watch = time.time()
            results = onnx_model(frame)
            stop_watch = time.time()
            elapsed_time = stop_watch - start_watch
            results = results[0]

            # Data being bundled as a packet
            result_packet = {
                "inference_time": elapsed_time,
                "detections": json.loads(yolo_to_json(results)),  # all detections
                "processed_frame": results.plot(),           # list of annotated images
                "verbose": [r.verbose() for r in results]                  # human-readable summary for each image
            }


            '''Below is for updated version of ultralytics'''
            # result_packet = {
            #     "inference_time": elapsed_time,
            #     "detections": [json.loads(results)],  # all detections
            #     "processed_frame": results.plot(),           # list of annotated images
            #     "verbose": [r.verbose() for r in results]                  # human-readable summary for each image
            # }

            if test_toggle: # Log inference outputs
                export_results_to_json(result_packet)
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

            print(f"{colors['olive_green']}Thread3: Detected objects in {elapsed_time} seconds {reset}")
            thread_lock.acquire()
            obj_res_queue.put(result_packet)
            output_stack.put(result_packet["detections"])
            thread_lock.release()
            print(f"{colors['olive_green']}Thread3: obj_res_queue size: {len(obj_res_queue)} {reset}") if test_toggle else None


            # If frame was passed as an argument, break out of the loop (i.e. only run once)
            if frame_given:
                break
    print(f"{colors['olive_green']}Thread3: ...Finished running object detection model. {reset}") if test_toggle else None
    return result_packet if frame_given else None


def export_results_to_json(results):
    results = results.copy()
    input_image_path = "obj_input.jpg" # Path to the input image that was captured
    output_path = "obj_output.json"
    success, encoded_img = cv2.imencode('.jpg', results["processed_frame"])
    if not success:
        print("ERROR: Failed to encode image", file=sys.stderr)
    with open(input_image_path, "wb") as f:
        f.write(encoded_img.tobytes())
    # Convert to base64
    results["processed_frame"] = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
    try:
        with open(output_path, "w") as f:
            json.dump(results, f)
    except Exception:
        print(f"{colors['olive_green']} ERROR: Failed to write output to file: {output_path} {reset}", file=sys.stderr)
    return

def log_inference_outputs(res):
    print(f"{colors['olive_green']}Thread3: results: {res} {reset}")
    return


def yolo_to_json(results, names=None):
    all_parsed = []

    # Handle batch or single-image
    is_batch = hasattr(results, '__iter__') and not isinstance(results, dict)
    result_list = results if is_batch else [results]

    for result in result_list:
        parsed = []
        boxes = result.boxes

        result_names = names or getattr(result, 'names', {i: str(i) for i in range(1000)})

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(box.cls)
            parsed.append({
                "name": result_names.get(cls_id, str(cls_id)),
                "class": cls_id,
                "confidence": float(box.conf),
                "box": {
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3])
                }
            })

        all_parsed.append(parsed)

    return json.dumps(all_parsed)
