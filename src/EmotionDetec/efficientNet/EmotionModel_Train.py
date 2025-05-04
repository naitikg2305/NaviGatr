import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

# Path to dataset
train_dir = '/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/Dataset/train'
test_dir = '/home/naitikg2305/ENEE408April24/NaviGatr/src/EmotionDetec/Dataset/test'

# Load and preprocess datasets
batch_size = 32
img_size = (224, 224)

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical',  # one-hot encoding for softmax
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Optional: prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# Load pretrained EfficientNet (ImageNet weights)
feature_extractor = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",  # use feature_vector, not classification
    input_shape=(224, 224, 3),
    trainable=False  # You can fine-tune later
)

# Full model
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(7, activation='softmax')  # FER has 7 emotion classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
epochs = 10  # or more depending on how long you want to train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
model.save("fer_emotion_model.h5")
