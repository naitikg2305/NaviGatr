import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load saved model
model = tf.keras.models.load_model("fer_emotion_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})

# Load image
img = cv2.imread("/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/efficientNet/20250312_225342.jpg")
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Only if needed
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
emotion_index = np.argmax(pred)
confidence = np.max(pred)

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print("Predicted emotion scores:", pred)
print(f"Predicted Emotion: {class_names[emotion_index]} ({confidence*100:.1f}%)")
