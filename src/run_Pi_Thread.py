import subprocess
import numpy as np
import cv2
import threading
import queue
import time
from objectDetection.main import run_objModel
from distance.get_depth import get_depth
from EmotionDetec.emotion_tflite import run_emotion_model_on_tpu
from text_to_speech import text_to_speech

# Queues for thread communication
obj_queue = queue.Queue()
depth_queue = queue.Queue()

# Function to capture image from libcamera
def capture_frame():
    result = subprocess.run([
        "libcamera-still", "-n", "-t", "100", "-o", "-", "--encoding", "jpg"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0 or len(result.stdout) == 0:
        print("Couldn't capture frame", result.stderr.decode())
        return

    img_array = np.frombuffer(result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        print("Couldn't decode frame")
        return
    else:
        print("frame returned")
        return frame


def run_object_model(frame):
    return run_objModel(frame)

def run_depth_model(frame):
    return get_depth(frame)["depth"]

def run_emotion_model(crop):    
    return run_emotion_model_on_tpu(crop)

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

    for s in(obj_results):
        box_array=[]
        for i in range(s["box"][0], s["box"][2]):            
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
        
        if name == "person":
            face_crop = frame[s["box"][1]:s["box"][3], s["box"][0]:s["box"][2]]
            emotion, confidence = run_emotion_model(face_crop)
            confidence_percentage = confidence*100
            boxes.append({"objname": name, "closest point": min(box_array), "Direction": direction, "Emotion": {emotion} , "Confidence": {confidence_percentage} })
        else:
            boxes.append({"objname": name, "closest point": min(box_array), "Direction": direction})



        

        
    print("\n--- Final Output ---")
    for box in boxes:
        if(box["objname"] == "person") :
            print(f"{box['objname']} {box['Emotion']} {box['Direction']} {box['closest point'][0]:.2f} meters away")
            text_to_speech(f"{box['objname']} {box['Emotion']} {box['Direction']} {box['closest point'][0]:.2f} meters away")
        else:
            print(f"{box['objname']} {box['Direction']} {box['closest point'][0]:.2f} meters away")
            text_to_speech(f"{box['objname']} {box['Direction']} {box['closest point'][0]:.2f} meters away")


            
if __name__ == "__main__":
    while True:
        process_frame_threaded()
        print("\n-- Waiting for next frame --\n")

        time.sleep(0.5)
