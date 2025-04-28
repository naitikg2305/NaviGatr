import sys
import cv2
import json
import base64
import numpy as np
from src.obj_detect.run_model import run_obj_detect_model

input_path = sys.argv[1]
output_path = sys.argv[2]

frame = cv2.imread(input_path)
if frame is None:
    print("ERROR: Failed to read image from file", file=sys.stderr)
    sys.exit(1)

result_packet = run_obj_detect_model(frame=frame, test_toggle=True)

try:
    result_img = result_packet["processed_frame"]  # numpy array with bboxes drawn

    # Show image using OpenCV
    print(f"Type of result_img: {type(result_img)}")
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)

except Exception:   
    print(f"ERROR: Failed to display image: {result_packet['processed_frame']}", file=sys.stderr)
    sys.exit(1)


''' No need to encode, don't ask me why'''
success, encoded_img = cv2.imencode('.jpg', result_img)


# Now encode
# success, encoded_img = cv2.imencode('.jpg', img_cv2)

if not success:
    print("ERROR: Failed to encode image", file=sys.stderr)
    sys.exit(1)



# Convert to base64
result_packet["processed_frame"] = base64.b64encode(encoded_img.tobytes()).decode('utf-8')

try:
    with open(output_path, "w") as f:
        json.dump(result_packet, f)

except Exception:
    print(f"ERROR: Failed to write output to file: {output_path}", file=sys.stderr)
    sys.exit(1)
