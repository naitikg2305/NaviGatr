import base64
import json
import sys
from PIL import Image
import cv2
import numpy as np
import torch
from src.distance.run_model import run_dep_detect_model

# ...
input_path = sys.argv[1]
output_path = sys.argv[2]

frame = cv2.imread(input_path)
if frame is None:
    print("ERROR: Failed to read image from file", file=sys.stderr)
    sys.exit(1)

result_packet = run_dep_detect_model(frame=input_path, test_toggle=False)

# Extract depth (torch.Tensor) and focal length
depth = result_packet["processed_frame"]
focallength_px = result_packet["focallength_px"]

# Handle depth tensor (convert to NumPy and then to list for JSON serialization)
if isinstance(depth, torch.Tensor):
    depth_np = depth.detach().cpu().numpy()
    result_packet["processed_frame"] = depth_np.tolist()  # Convert to list (JSON serializable)

# (Optional) If you need to visualize the depth map:
# Normalize the depth to a range [0, 255] for visualization
depth_vis = (depth_np / depth_np.max() * 255).astype(np.uint8)
depth_vis_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# Encode the visualized depth map as an image (for saving)
success, encoded_img = cv2.imencode('.jpg', depth_vis_colored)
if not success:
    print("ERROR: Failed to encode depth image", file=sys.stderr)
    sys.exit(1)

# Add the visualized depth image (optional)
result_packet["depth_image"] = base64.b64encode(encoded_img.tobytes()).decode("utf-8")

# Ensure `focallength_px` is JSON serializable (if it's a tensor or something else)
if isinstance(focallength_px, torch.Tensor):
    result_packet["focallength_px"] = focallength_px.item()  # Convert tensor to scalar

# Save sanitized JSON
with open(output_path, "w") as f:
    json.dump(result_packet, f)
