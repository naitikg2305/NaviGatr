from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size and batch size
img_width, img_height = 48, 48
batch_size = 32

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data generator (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    '/home/naitikg2305/ENEE408Capstone/NaviGatr/src/EmotionDetec/Dataset/train',
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

# Validation data generator
validation_generator = validation_datagen.flow_from_directory(
    '/home/naitikg2305/ENEE408Capstone/NaviGatr/src/EmotionDetec/Dataset/test',
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)
