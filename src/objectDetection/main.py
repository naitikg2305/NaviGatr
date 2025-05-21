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


def load_labels_by_line(path):
    labels = {}
    with open(path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            label = line.strip()
            if label and label.lower() != 'n/a':
                labels[idx] = label
    return labels

labels = load_labels_by_line(LABELS_PATH)
def capture_and_show_cv2():
    result = subprocess.run([
        "libcamera-still", "-n", "-t", "100", "-o", "-", "--encoding", "jpg"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0 or len(result.stdout) == 0:
        print("[ERROR] Failed to capture image:", result.stderr.decode())
        return

    img_array = np.frombuffer(result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        print("[ERROR] Failed to decode image")
        return
    else:
        print("frame returned")
        return frame


def run_objModel(frame):

    
    original_image = frame



    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    print("interpreter loaded with:" + str(interpreter._delegates))
    
    input_size = common.input_size(interpreter)
    print("input size",input_size)
    resized_image = cv2.resize(original_image, input_size)

   
    input_tensor = resized_image.flatten().astype(np.uint8)
    run_inference(interpreter, input_tensor)

  
    objs = detect.get_objects(interpreter, score_threshold=0.4)

    # Prepare blank mask with shape of original image
    height, width = original_image.shape[:2]
    mask_array = np.zeros((height, width), dtype=np.uint8)

    # Calculate scaling factors from model input to original image size
    scale_x = width / input_size[0]
    scale_y = height / input_size[1]

    # Draw detection boxes onto the mask
    for idx, obj in enumerate(objs, start=1):
        bbox = obj.bbox
        x0, y0 = int(bbox.xmin * scale_x), int(bbox.ymin * scale_y)
        x1, y1 = int(bbox.xmax * scale_x), int(bbox.ymax * scale_y)
        cv2.rectangle(mask_array, (x0, y0), (x1, y1), color=idx, thickness=-1)
    return_list =[]
    for obj in objs:
        return_list.append({"object": labels.get(obj.id, obj.id), "box": [int(obj.bbox.xmin*scale_x), int(obj.bbox.ymin*scale_y), int(obj.bbox.xmax*scale_x), int(obj.bbox.ymax*scale_y)]})
        print(f"Detected: {labels.get(obj.id, obj.id)} with score {obj.score:.2f} at {[int(obj.bbox.xmin*scale_x), int(obj.bbox.ymin*scale_y), int(obj.bbox.xmax*scale_x), int(obj.bbox.ymax*scale_y)]}")

    return return_list





