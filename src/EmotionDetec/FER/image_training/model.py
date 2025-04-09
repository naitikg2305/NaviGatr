import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Path to the test image
img_path = "/home/naitikg2305/ENEE408Capstone/NaviGatr/src/EmotionDetec/efficientNet/20250312_225342.jpg"  # Replace with your image file path

# Load image in grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Resize to 48x48
img = cv2.resize(img, (48, 48))

# Normalize and reshape
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)     # (1, 48, 48)
img = np.expand_dims(img, axis=-1)    # (1, 48, 48, 1)

# Predict
prediction = model.predict(img)
predicted_class = emotion_labels[np.argmax(prediction)]

print(f"Predicted Emotion: {predicted_class}")
