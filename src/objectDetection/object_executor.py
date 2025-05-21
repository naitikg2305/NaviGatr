from pycoral.utils import edgetpu
from pycoral.adapters import common, detect
import cv2
import numpy as np
import os

object_interpreter = size = labels = None

def init_object_executor():
    import os
    global object_interpreter, size, labels
    
    OBJECT_MODEL_FILE = f"{os.path.dirname(__file__)}/coral_models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    OBJECT_LABELS_FILE = f"{os.path.dirname(__file__)}/coral_models/coco_labels.txt"

    if not (os.path.exists(OBJECT_MODEL_FILE) and os.path.exists(OBJECT_LABELS_FILE)):
        raise FileNotFoundError("Object model tflite file or class labels file missing...")
    
    object_interpreter = edgetpu.make_interpreter(OBJECT_MODEL_FILE)
    object_interpreter.allocate_tensors()

    size = common.input_size(object_interpreter)

    with open(OBJECT_LABELS_FILE, "r") as labels_file:
        labels = {idx : label for idx, label in enumerate(labels_file.read().split("\n")) if label != "n/a" and label != ""}
        

def get_objects(frame):
    global object_interpreter

    img_array = np.frombuffer(frame, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    resized = cv2.resize(frame, size)
    input_tensor = resized.flatten().astype(np.uint8)
    edgetpu.run_inference(object_interpreter, input_tensor)

    objs = detect.get_objects(object_interpreter, score_threshold=0.4)

    height, width = frame.shape[:2]
    scale_x = width / size[0]
    scale_y = height / size[1]

    object_locations = []
    
    for obj in objs:
        object_locations.append({"label": labels.get(obj.id, obj.id), "box": [int(obj.bbox.xmin*scale_x), int(obj.bbox.ymin*scale_y), int(obj.bbox.xmax*scale_x), int(obj.bbox.ymax*scale_y)]})
        print(f"Detected: {labels.get(obj.id, obj.id)} with score {obj.score:.2f} at {[int(obj.bbox.xmin*scale_x), int(obj.bbox.ymin*scale_y), int(obj.bbox.xmax*scale_x), int(obj.bbox.ymax*scale_y)]}")

    return object_locations


