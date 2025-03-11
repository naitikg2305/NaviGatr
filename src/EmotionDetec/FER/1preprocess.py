import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('/home/naitikg2305/ENEE408Capstone/NaviGatr/src/EmotionDetec/Dataset/data.zip')

# Extract features and labels
pixels = data['pixels'].tolist()
width, height = 48, 48
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height)
    faces.append(face.astype('float32'))
faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)

emotions = pd.get_dummies(data['emotion']).values

# Normalize pixel values
faces /= 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)
