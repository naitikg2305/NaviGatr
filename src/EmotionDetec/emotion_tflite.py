import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter, run_inference
from pycoral.adapters import common


EMOTION_MODEL_PATH = "/home/navigatr/enee408NFRFRfinal/NaviGatr/src/EmotionDetec/fer_emotion_model_int8_edgetpu.tflite"
emotion_interpreter = make_interpreter(EMOTION_MODEL_PATH)
emotion_interpreter.allocate_tensors()


emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()


class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print("Expected input shape:", emotion_input_details[0]['shape'])


def run_emotion_model_on_tpu(face_crop):
    input_shape = common.input_size(emotion_interpreter)
    resized = cv2.resize(face_crop, input_shape)

    resized = cv2.resize(face_crop, input_shape)
    input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
    flat_input = input_tensor.flatten()

    

    
    run_inference(emotion_interpreter, flat_input)


    output = emotion_interpreter.tensor(emotion_output_details[0]['index'])()[0]
    emotion_index = np.argmax(output)
    confidence = output[emotion_index] / 255.0 if emotion_output_details[0]['dtype'] == np.uint8 else output[emotion_index]

    print(f"Detected Emotion: {class_names[emotion_index]} ({confidence*100:.1f}%)")
    return class_names[emotion_index], confidence

