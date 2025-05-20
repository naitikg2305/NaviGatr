import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter, run_inference
from pycoral.adapters import common

# Path to Edge TPU model
EMOTION_MODEL_PATH = "/home/navigatr/enee408NFRFRfinal/NaviGatr/src/EmotionDetec/efficientNet/fer_emotion_model_int8_edgetpu.tflite"
emotion_interpreter = make_interpreter(EMOTION_MODEL_PATH)
emotion_interpreter.allocate_tensors()

# Get input/output details
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# FER2013 class names
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print("Expected input shape:", emotion_input_details[0]['shape'])


def run_emotion_model_on_tpu(face_crop):
    # Convert BGR to grayscale
    face_crop = cv2.imread(face_crop)
    # gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # Resize to model input shape (assumes square input)
    input_shape = common.input_size(emotion_interpreter)
    resized = cv2.resize(face_crop, input_shape)

    # Expand to (1, height, width, 1) and convert to uint8
    resized = cv2.resize(face_crop, input_shape)
    input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
    flat_input = input_tensor.flatten()

    

    # Set input and invoke
    # common.set_input(emotion_interpreter, input_tensor)
    run_inference(emotion_interpreter, flat_input)

    # Get results
    output = emotion_interpreter.tensor(emotion_output_details[0]['index'])()[0]
    emotion_index = np.argmax(output)
    confidence = output[emotion_index] / 255.0 if emotion_output_details[0]['dtype'] == np.uint8 else output[emotion_index]

    print(f"Detected Emotion: {class_names[emotion_index]} ({confidence*100:.1f}%)")
    return class_names[emotion_index], confidence



run_emotion_model_on_tpu("/home/navigatr/enee408NFRFRfinal/NaviGatr/src/EmotionDetec/efficientNet/20250519_222355.jpg")