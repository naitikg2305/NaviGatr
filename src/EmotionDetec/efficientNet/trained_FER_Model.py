import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load saved model
model = tf.keras.models.load_model("fer_emotion_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})

# Load image
image_path = '/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/efficientNet/Disgust.jpg'  # Update this with an actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Make a prediction


pred = model.predict(image)
emotion_index = np.argmax(pred)
confidence = np.max(pred)
print("Disgust \n")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"Predicted Emotion: {class_names[emotion_index]} ({confidence*100:.1f}%)")


image_path = '/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/efficientNet/Happy.jpg'  # Update this with an actual image path
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Make a prediction


pred = model.predict(image)
emotion_index = np.argmax(pred)
confidence = np.max(pred)
print("Happy \n")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"Predicted Emotion: {class_names[emotion_index]} ({confidence*100:.1f}%)")


image_path = '/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/efficientNet/Neutral.jpg'  # Update this with an actual image path
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Make a prediction


pred = model.predict(image)
emotion_index = np.argmax(pred)
confidence = np.max(pred)
print("Neutral \n")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"Predicted Emotion: {class_names[emotion_index]} ({confidence*100:.1f}%)")



image_path = '/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/efficientNet/Sad.jpg'  # Update this with an actual image path
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Make a prediction


pred = model.predict(image)
emotion_index = np.argmax(pred)
confidence = np.max(pred)
print("Sad \n")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"Predicted Emotion: {class_names[emotion_index]} ({confidence*100:.1f}%)")




print("\n -----------------------------------------------------\n")


image_path = '/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/efficientNet/cropped_Neutral.png'  # Update this with an actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Make a prediction


pred = model.predict(image)
emotion_index = np.argmax(pred)
confidence = np.max(pred)
print("Cropped Neutral \n")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"Predicted Emotion: {class_names[emotion_index]} ({confidence*100:.1f}%)")


image_path = '/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/efficientNet/cropped_Happy.png'  # Update this with an actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Make a prediction


pred = model.predict(image)
emotion_index = np.argmax(pred)
confidence = np.max(pred)
print("Cropped Happy \n")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"Predicted Emotion: {class_names[emotion_index]} ({confidence*100:.1f}%)")


image_path = '/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/efficientNet/cropped_Sad.png'  # Update this with an actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Make a prediction


pred = model.predict(image)
emotion_index = np.argmax(pred)
confidence = np.max(pred)
print("Cropped Sad \n")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"Predicted Emotion: {class_names[emotion_index]} ({confidence*100:.1f}%)")


image_path = '/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/efficientNet/cropped_Disgust.png'  # Update this with an actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Make a prediction


pred = model.predict(image)
emotion_index = np.argmax(pred)
confidence = np.max(pred)
print("Cropped Disgust \n")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"Predicted Emotion: {class_names[emotion_index]} ({confidence*100:.1f}%)")