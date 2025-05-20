from ultralytics import YOLO
import numpy as np
from cv2 import resize, imshow, imread, imwrite
# Load a model
model = YOLO("/home/navigatr/enee408NFRFRfinal/NaviGatr/src/obj_detect/yolo11n_int8.tflite")  # Load an official model or custom model
img = imread("/home/navigatr/enee408NFRFRfinal/NaviGatr/src/test_output.jpg")
img = resize(img, (640, 640))
imwrite("resized.jpg", img)
# input_tensor = img.astype(np.uint8)
# input_tensor = np.expand_dims(input_tensor, axis=0)

# Run Prediction
model.predict(img)