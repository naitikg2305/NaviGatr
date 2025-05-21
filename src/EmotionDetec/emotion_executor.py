from pycoral.utils import edgetpu
from pycoral.adapters import common
import cv2
import numpy as np

emotion_interpreter = size = emotion_input_details = emotion_output_details = None
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def init_emotion_executor():
    import os
    global emotion_interpreter, size, emotion_input_details, emotion_output_details
    
    EMOTION_MODEL_FILE = f"{os.path.dirname(__file__)}/fer_emotion_model_int8_edgetpu.tflite"

    if not os.path.exists(EMOTION_MODEL_FILE):
        raise FileNotFoundError("Emotion model tflite file missing...")
    
    emotion_interpreter = edgetpu.make_interpreter(EMOTION_MODEL_FILE)
    emotion_interpreter.allocate_tensors()

    size = common.input_size(emotion_interpreter)

    emotion_input_details = emotion_interpreter.get_input_details()
    emotion_output_details = emotion_interpreter.get_output_details()

def get_emotion(bounding_box):
    global emotion_interpreter

    resized = cv2.resize(bounding_box, size)

    input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
    flat_input = input_tensor.flatten()

    edgetpu.run_inference(emotion_interpreter, flat_input)

    output = emotion_interpreter.tensor(emotion_output_details[0]['index'])()[0]
    emotion_index = np.argmax(output)
    
    confidence = output[emotion_index] / 255.0 if emotion_output_details[0]['dtype'] == np.uint8 else output[emotion_index]

    print(f"Detected Emotion: {emotion_labels[emotion_index]} ({confidence*100:.1f}%)")
    return emotion_labels[emotion_index], confidence

