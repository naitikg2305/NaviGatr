import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import time 

start=time.time()
# Load EfficientNet-Lite feature vector model
feature_extractor_layer = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2",
    trainable=False
)

# Ensure it's wrapped inside a model correctly
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),  # Explicit Input Layer
    tf.keras.layers.Lambda(lambda x: feature_extractor_layer(x)),  # Wrap Hub Layer in Lambda
    tf.keras.layers.Dense(7, activation='softmax')  # Change this based on your dataset
])

# Load an image
image_path = '/home/naitikg2305/ENEE408Capstone/NaviGatr/src/EmotionDetec/efficientNet/photos/20250312_225342.jpg'  # Update this with an actual image path
image1_image = cv2.imread(image_path)
image1 = cv2.resize(image1_image, (224, 224))
image1 = np.array(image1) / 255.0
image1 = np.expand_dims(image1, axis=0)

image_path = '/home/naitikg2305/ENEE408Capstone/NaviGatr/src/EmotionDetec/efficientNet/20250312_225343.jpg'  # Update this with an actual image path
image2_image = cv2.imread(image_path)
image2 = cv2.resize(image2_image, (224, 224))
image2 = np.array(image2) / 255.0
image2 = np.expand_dims(image2, axis=0)

image_path = '/home/naitikg2305/ENEE408Capstone/NaviGatr/src/EmotionDetec/efficientNet/20250312_225346.jpg'  # Update this with an actual image path
image3_image = cv2.imread(image_path)
image3 = cv2.resize(image3_image, (224, 224))
image3 = np.array(image3) / 255.0
image3 = np.expand_dims(image3, axis=0)

image_path = '/home/naitikg2305/ENEE408Capstone/NaviGatr/src/EmotionDetec/efficientNet/20250312_225347.jpg'  # Update this with an actual image path
image4_image = cv2.imread(image_path)
image4 = cv2.resize(image4_image, (224, 224))
image4 = np.array(image4) / 255.0
image4 = np.expand_dims(image4, axis=0)

# Make a prediction
predictions = model.predict(image1)
print("Predicted emotion scores for image1:", predictions)

predictions = model.predict(image2)
print("Predicted emotion scores for image2:", predictions)

predictions = model.predict(image3)
print("Predicted emotion scores for image3:", predictions)

predictions = model.predict(image4)
print("Predicted emotion scores for image4:", predictions)

end = time.time()
print(f"Inference Time: {end - start:.2f}s")


