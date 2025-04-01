
import cv2
import numpy as np
from openvino import Core

"""
# Paths to model files
model_xml = r"C:\Users\mewri\OneDrive\Documents\ENEE408N\nanodet-plus-m_416_openvino.xml"
model_bin = r"C:\Users\mewri\OneDrive\Documents\ENEE408N\nanodet-plus-m_416_openvino.bin"
image_path = r"C:\Users\mewri\OneDrive\Documents\ENEE408N\COCO_train2014_000000196639.jpg"  # Path to the image you want to test

# Load OpenVINO model
ie = Core()
model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Get input and output layer
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Load image
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

# Preprocess image: Resize, normalize, and change to CHW format
input_img = cv2.resize(image, (416, 416))  # Resize to 416x416
input_img = input_img.transpose(2, 0, 1)  # HWC → CHW
input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
input_img = input_img.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Run inference
results = compiled_model([input_img])[output_layer]

# Process the results: The output format depends on the model's implementation, so we assume it has bounding boxes and class ids
for detection in results[0]:
    x1, y1, x2, y2 = detection[:4]  # Extract bounding box
    score = detection[4]  # Confidence score
    class_id = int(detection[5])  # Class ID
    
    # Filter detections based on confidence score
    if score > 0.3:  # Confidence threshold
        # Scale bounding box back to original image size
        x1, y1, x2, y2 = int(x1 * image_width), int(y1 * image_height), int(x2 * image_width), int(y2 * image_height)
        
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add the label text (Class ID and confidence)
        label = f"Class {class_id}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the result
cv2.imshow("Detection Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

# Paths to model files
model_xml = r"C:\Users\mewri\OneDrive\Documents\ENEE408N\nanodet-plus-m_416_openvino.xml"
model_bin = r"C:\Users\mewri\OneDrive\Documents\ENEE408N\nanodet-plus-m_416_openvino.bin"
image_path = r"C:\Users\mewri\OneDrive\Documents\ENEE408N\COCO_train2014_000000196639.jpg"

# Load OpenVINO model
ie = Core()
model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Get input and output layer
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Load image
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

# Preprocess image: Resize, normalize, and change to CHW format
input_img = cv2.resize(image, (416, 416))  # Resize to 416x416
input_img = input_img.transpose(2, 0, 1)  # HWC → CHW
input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
input_img = input_img.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Run inference
results = compiled_model([input_img])[output_layer]

# Debug: Print output shape and some details from the results
print("Results shape:", results.shape)
print("First detection details (if any):", results[0][0])  # Print the first detection to check if there are any

# Fake bounding box (for testing visibility)
fake_x1, fake_y1, fake_x2, fake_y2 = 50, 50, 350, 350
fake_score = 1.0  # Fake high score to display it
fake_class_id = 0  # Fake class ID

# Draw the fake bounding box
cv2.rectangle(image, (fake_x1, fake_y1), (fake_x2, fake_y2), (0, 255, 0), 2)
label = f"Fake Class {fake_class_id}: {fake_score:.2f}"
cv2.putText(image, label, (fake_x1, fake_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the result with the fake bounding box
cv2.imshow("Detection Results with Fake Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
