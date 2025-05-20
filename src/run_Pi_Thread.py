import subprocess
import numpy as np
import cv2
import threading
import json
import base64
import tensorflow as tf
import tensorflow_hub as hub
import queue
import time
from pi_Files.objectDetection.main import run_objModel
from distance.run_model import run_dep_detect_model
from distance.get_depth import get_depth

# Load the emotion model once
# model = tf.keras.models.load_model("fer_emotion_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})
interpreter = tf.lite.Interpreter(model_path="fer_emotion_model.tflite")
interpreter.allocate_tensors()

# Class labels for the emotion model
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Queues for thread-safe results
obj_queue = queue.Queue()
depth_queue = queue.Queue()

# Function to capture image from libcamera

def capture_frame():
    result = subprocess.run([
        "libcamera-still", "-n", "-t", "100", "-o", "-", "--encoding", "jpg"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0 or len(result.stdout) == 0:
        print("[ERROR] Failed to capture image:", result.stderr.decode())
        return

    img_array = np.frombuffer(result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        print("[ERROR] Failed to decode image")
        return
    else:
        print("frame returned")
        return frame

# Dummy object and depth model functions (replace with actual logic)
def run_object_model(frame):
    return run_objModel(frame)
    

def run_depth_model(frame):
    return get_depth(frame)["depth"]

# def run_emotion_model(crop):
#     face_resized = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
#     face_input = np.expand_dims(face_resized, axis=0)
#     prediction = model.predict(face_input)
#     return class_names[np.argmax(prediction)], float(np.max(prediction))

def object_worker(frame):
    obj_results = run_object_model(frame)
    obj_queue.put(obj_results)

def depth_worker(frame):
    depth_map = run_depth_model(frame)
    depth_queue.put(depth_map)

def process_frame_threaded():
    frame = capture_frame()
    frame
    if frame is None:
        return

    # Start model threads
    t1 = threading.Thread(target=object_worker, args=(frame,))
    t2 = threading.Thread(target=depth_worker, args=(frame,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    obj_results = obj_queue.get()
    depth_map = depth_queue.get()

    clockWidth = np.size(depth_map[0])
    clockQuad = clockWidth / 4
    boxes = []

    # for obj in obj_results:
    #     name = obj['name']
    #     x1, y1, x2, y2 = obj['box']['x1'], obj['box']['y1'], obj['box']['x2'], obj['box']['y2']
    #     box_array = []

    #     for i in (x1, x2):
    #         for j in (y1, y2):
    #             box_array.append([depth_map[j][i]])

    #     cx = (x1 + x2) / 2
    # print(obj_results)
    # print(obj_results[0])
    # print(obj_results[0]["box"])
    # print(type(obj_results[0]["box"]))
    # print(obj_results[0]["box"])
    # print("x1",obj_results[0]["box"][0])
    # print("y1", obj_results[0]["box"][1])
    # print("x2", obj_results[0]["box"][2])
    # print("x3", obj_results[0]["box"][3])
    for s in(obj_results):
    # for s in (obj_output):
        box_array=[]
        for i in range(s["box"][0], s["box"][2]):
        #for i in  (int(s['box']['x1']), int(s['box']['x2'])):
            

            name = s['object']

            for j in range(s["box"][1], s["box"][3]):
                x = []
                x.append(depth_map[j][i])
                box_array.append(x)

        cx = (s["box"][0]+ s["box"][2])/2

        if cx < clockQuad:
            direction = "10 Oclock" if cx < clockQuad / 2 else "11 Oclock"
        elif cx < clockQuad * 2:
            direction = "11 Oclock" if cx < clockQuad * 1.5 else "12 Oclock"
        elif cx < clockQuad * 3:
            direction = "12 Oclock" if cx < clockQuad * 2.5 else "1 Oclock"
        else:
            direction = "1 Oclock" if cx < clockQuad * 3.5 else "2 Oclock"
        boxes.append({"objame": name, "closest point": min(box_array), "Direction": direction})
        # if name == "person":
        #     face_crop = frame[s["box"][1]:s["box"][3], s["box"][0]:s["box"][2]]
        #     emotion, confidence = run_emotion_model(face_crop)
        #     confidence_percentage = confidence*100
        #     boxes.append({"objame": name, "closest point": min(box_array), "Direction": direction, "Emotion": {emotion} , "Confidence": {confidence_percentage} })
        # else:
        #     boxes.append({"objame": name, "closest point": min(box_array), "Direction": direction})



        

        
    print("\n--- Final Output ---")
    for box in boxes:
        print(f"{box['objame']} {box['Direction']} {box['closest point'][0]:.2f} meters away")
        # if(box["objname"] == "person") :
        #     print(f"{box['objame']} {box["Emotion"]} {box['Direction']} {box['closest point'][0]:.2f} meters away")

        # else:
        #     print(f"{box['objame']} {box['Direction']} {box['closest point'][0]:.2f} meters away")


            
if __name__ == "__main__":
    while True:
        process_frame_threaded()
        print("\n--- Waiting for next frame ---\n")
        # Optional delay
        time.sleep(0.5)
