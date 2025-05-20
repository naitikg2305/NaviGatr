import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter, run_inference
from pycoral.adapters import common, detect
import jax 
from io import BytesIO
import subprocess

# Load model and labels
MODEL_PATH = '/home/navigatr/enee408NFRFRfinal/NaviGatr/src/objectDetection/coral_models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
LABELS_PATH = '/home/navigatr/enee408NFRFRfinal/NaviGatr/src/objectDetection/coral_models/coco_labels.txt'
IMAGE_PATH = '/home/navigatr/enee408NFRFRfinal/NaviGatr/src/pi_Files/objectDetection/20250518_120156.jpg'

# Load labels
def load_labels_by_line(path):
    labels = {}
    with open(path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            label = line.strip()
            if label and label.lower() != 'n/a':
                labels[idx] = label
    return labels

# Load your labels
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

    # cv2.imshow("Captured Image", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def run_objModel(frame):

    #print(f"Loaded {len(labels)} labels")  # You should see something like 80
    # Load image using OpenCV (BGR by default)
    # image = np.frombuffer(frame, dtype=np.uint8)
    # original_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #original_image = cv2.imread(IMAGE_PATH)
    # original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = frame
    # original_image = cv2.imread(frame)

    # Set up interpreter first
    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    print("interpreter loaded with:" + str(interpreter._delegates))
    # Resize to model input size (after interpreter is ready)
    input_size = common.input_size(interpreter)
    print("input size",input_size)
    resized_image = cv2.resize(original_image, input_size)

    # Flatten and convert to uint8 before running inference
    input_tensor = resized_image.flatten().astype(np.uint8)
    run_inference(interpreter, input_tensor)

    # Get detection results
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
    # Print detection results
    # print(objs[0].bbox.xmin)
    for obj in objs:
        return_list.append({"object": labels.get(obj.id, obj.id), "box": [int(obj.bbox.xmin*scale_x), int(obj.bbox.ymin*scale_y), int(obj.bbox.xmax*scale_x), int(obj.bbox.ymax*scale_y)]})
        print(f"Detected: {labels.get(obj.id, obj.id)} with score {obj.score:.2f} at {[int(obj.bbox.xmin*scale_x), int(obj.bbox.ymin*scale_y), int(obj.bbox.xmax*scale_x), int(obj.bbox.ymax*scale_y)]}")

    # for obj in objs:
    #     bbox = obj.bbox
    #     x0, y0 = int(bbox.xmin * scale_x), int(bbox.ymin * scale_y)
    #     x1, y1 = int(bbox.xmax * scale_x), int(bbox.ymax * scale_y)

    #     # Get label name and score
    #     label = labels.get(obj.id, f"ID {obj.id}")
    #     score = f"{obj.score:.2f}"
    #     label_text = f"{label} ({score})"

    #     # Draw bounding box
    #     cv2.rectangle(original_image, (x0, y0), (x1, y1), (0, 255, 0), 2)

    #     # Calculate text size for background box
    #     (text_width, text_height), baseline = cv2.getTextSize(label_text,
    #                                                         cv2.FONT_HERSHEY_SIMPLEX,
    #                                                         0.5, 1)

    #     # Draw background rectangle for text
    #     cv2.rectangle(original_image,
    #                 (x0, y0 - text_height - 6),
    #                 (x0 + text_width + 4, y0),
    #                 (0, 255, 0), -1)

    #     # Put label text on the image
    #     cv2.putText(original_image, label_text, (x0 + 2, y0 - 4),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # output_path = "detections.jpg"
    # cv2.imwrite(output_path, original_image)
    # print(f"Saved detection image to: {output_path}")
    # print(jax.devices())
        

    # Save the mask for visualization (scale values for contrast)
    return return_list







# run_objModel(capture_and_show_cv2())
# run_objModel(IMAGE_PATH)

    # Draw bounding boxes and labels on the image
    
