import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter, run_inference
from pycoral.adapters import common, detect
import jax 
from io import BytesIO
import subprocess

MODEL_PATH = '/home/navigatr/enee408NFRFRfinal/NaviGatr/src/objectDetection/coral_models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
LABELS_PATH = '/home/navigatr/enee408NFRFRfinal/NaviGatr/src/objectDetection/coral_models/coco_labels.txt'
IMAGE_PATH = '/home/navigatr/enee408NFRFRfinal/NaviGatr/src/pi_Files/objectDetection/20250518_120156.jpg'

interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

input_size = common.input_size(interpreter)

def load_labels_by_line(path):
    labels = {}
    with open(path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            label = line.strip()
            if label and label.lower() != 'n/a':
                labels[idx] = label
    return labels

labels = load_labels_by_line(LABELS_PATH)

def run_objModel(frame):    
    original_image = frame

    resized_image = cv2.resize(original_image, input_size)
   
    input_tensor = resized_image.flatten().astype(np.uint8)
    run_inference(interpreter, input_tensor)

    objs = detect.get_objects(interpreter, score_threshold=0.4)

    height, width = original_image.shape[:2]

    # Scaling factor to convert between model and original image sizes
    scale_x = width / input_size[0]
    scale_y = height / input_size[1]

    return_list =[]
    for obj in objs:
        return_list.append({"object": labels.get(obj.id, obj.id), "box": [int(obj.bbox.xmin*scale_x), int(obj.bbox.ymin*scale_y), int(obj.bbox.xmax*scale_x), int(obj.bbox.ymax*scale_y)]})
        print(f"Detected: {labels.get(obj.id, obj.id)} with score {obj.score:.2f} at {[int(obj.bbox.xmin*scale_x), int(obj.bbox.ymin*scale_y), int(obj.bbox.xmax*scale_x), int(obj.bbox.ymax*scale_y)]}")

    return return_list





