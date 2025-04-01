from tensorflow.keras.optimizers import Adam
from CNN import *

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
