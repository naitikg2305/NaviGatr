import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

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
image_path = '/home/naitikg2305/ENEE408Capstone/NaviGatr/src/EmotionDetec/efficientNet/20250312_225342.jpg'  # Update this with an actual image path
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Make a prediction
predictions = model.predict(image)
print("Predicted emotion scores:", predictions)
