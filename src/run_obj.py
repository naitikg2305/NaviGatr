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

# cv2.imshow("Processed Frame", result_packet["processed_frame"])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

success, encoded_img = cv2.imencode('.jpg', result_packet["processed_frame"])
if not success:
    print("ERROR: Failed to encode image", file=sys.stderr)
    sys.exit(1)

result_packet["processed_frame"] = base64.b64encode(encoded_img.tobytes()).decode('utf-8')

with open(output_path, "w") as f:
    json.dump(result_packet, f)
