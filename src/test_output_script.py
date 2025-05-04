import base64
import json
import os
import subprocess
import threading
import time
import cv2
import numpy as np
import torch
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
#from src.webcam import capture_one_frame, connect_to_webcam, run_camera_service, view_processed_frames
#from src.sharable_data import obj_res_queue, depth_res_queue, emot_res_queue

# Load the Emotion Model, add this to the main script not here because we dont want it to load it over and over again 
# add this before the threads even start running, i.e. in the main thread at the start so it stays loaded
model = tf.keras.models.load_model("fer_emotion_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})



obj_output_json_path = "/home/naitikg2305/ENEE408April24/NaviGatr/obj_output.json" # Path to where output data will be stored
dep_output_json_path = "/home/naitikg2305/ENEE408April24/NaviGatr/dep_output.json" 

with open(obj_output_json_path, "r") as f:
        result_packet_obj = json.load(f)

    
        
processed_img_bytes = base64.b64decode(result_packet_obj["processed_frame"])
processed_img_np = np.frombuffer(processed_img_bytes, dtype=np.uint8)
processed_img = cv2.imdecode(processed_img_np, cv2.IMREAD_COLOR)
cv2.imshow("Processed Image", processed_img_np)
cv2.imshow(result_packet_obj["processed_frame"], processed_img)


#print(result_packet_obj)
#print(processed_img_bytes)
#print(processed_img_np)
#print(processed_img)
#result_packet_obj["processed_frame"] = processed_img
obj_results = result_packet_obj["detections"]
print(obj_results)
    #print(type(obj_results))
        # push processed frame to queue
    #obj_res_queue.append(result_packet_obj)


with open(dep_output_json_path, "r") as f:
    result_packet = json.load(f)
    
depth_results = result_packet["processed_frame"]

def getOutput(obj_output, depth_output):
    
    
    #print(depth_results[0][0])
    #print(type(depth_results))
    #print(np.size(depth_results[0]))


    a = 0 
    b = 0 

    boxes =[]

    #print(type(obj_results[0][1]['box']))
    #print(type(depth_results[1][1]))
    #print(np.size(depth_results[0]))
    # print(depth_results[1918][1061])

    print("------------------------------/n /n")
    print(obj_results[0])
    print (np.size(depth_results[0]))



    clockWidth = np.size(depth_results[0])
    clockQuad = clockWidth/4


    for s in(obj_results[0]):
        box_array=[]
        name = s['name']
        if name == "person":
            x1 = int(s['box']['x1'])
            y1 = int(s['box']['y1'])
            x2 = int(s['box']['x2'])
            y2 = int(s['box']['y2'])

            # Crop from the original image
            face_crop = processed_img[y1:y2, x1:x2]

            # Resize and preprocess for emotion model
            face_resized = cv2.resize(face_crop, (224, 224))
            face_input = face_resized.astype(np.float32) / 255.0
            face_input = np.expand_dims(face_input, axis=0)

            # Predict emotion
            prediction = model.predict(face_input)
            emotion_index = np.argmax(prediction)
            confidence = np.max(prediction)

            class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            print(f"Person detected at {x1},{y1} - {x2},{y2}: Emotion = {class_names[emotion_index]} ({confidence*100:.1f}%)")

        for i in  (int(s['box']['x1']), int(s['box']['x2'])):
            name = s['name']

            for j in (int(s['box']['y1']), int(s['box']['y2'])):
                x = []

                x.append(depth_results[j][i])
                box_array.append(x)

        

        boxXCenter = (int(s['box']['x1']) + int(s['box']['x2']))/2
        if(boxXCenter < clockQuad):
             if(boxXCenter < clockQuad/2):
                  boxes.append({"objame": name, 'closest point': min(box_array), 'Direction': "10 Oclock"})
             else:
                  boxes.append({"objame": name, 'closest point': min(box_array), 'Direction': "11 Oclock"})
                  
             
        elif(boxXCenter < clockQuad*2):
                if(boxXCenter < clockQuad):
                    boxes.append({"objame": name, 'closest point': min(box_array), 'Direction': "11 Oclock"})
                else:
                    boxes.append({"objame": name, 'closest point': min(box_array), 'Direction': "12 Oclock"})
             
        elif(boxXCenter < clockQuad*3):
                if(boxXCenter < clockQuad*1.5):
                    boxes.append({"objame": name, 'closest point': min(box_array), 'Direction': "12 Oclock"})
                else:
                    boxes.append({"objame": name, 'closest point': min(box_array), 'Direction': "1 Oclock"})
             
        elif(boxXCenter < clockQuad*4):
                if(boxXCenter < clockQuad*2):
                    boxes.append({"objame": name, 'closest point': min(box_array), 'Direction': "1 Oclock"})
                else:
                    boxes.append({"objame": name, 'closest point': min(box_array), 'Direction': "2 Oclock"})
             
        else:
            print("Error: Box center out of bounds")
        

    print("----------------------------------------- \n \n \n")
    print(boxes)
    print(min(box_array))



    for s in(boxes):
        print(( str(s["objame"]) + " "+ str(s["Direction"])+ " " + str(s["closest point"][0]) + " meters away" ))
        print("\n")
    # for i in range(0, 831):  
    # for i in range(831, 1918):
        

    #     for j in range(4, 1061):
    #         x = []
    #         x.append(depth_results[j][i])
    #         box_array.append(x)

    
    #print("SUCCESS: Received data back from run_depth.py")
    # Decode image and display
    #processed_img_bytes = base64.b64decode(result_packet["depth_image"])
    #processed_img_np = np.frombuffer(processed_img_bytes, dtype=np.uint8)
    #processed_img = cv2.imdecode(processed_img_np, cv2.IMREAD_COLOR)

    #print(result_packet)
    #print(processed_img_bytes)
    #print(processed_img_np)
    #print(processed_img)

    #result_packet["processed_frame"] = processed_img

        # push processed frame to queue



        # Find relevant objects and determine depth to objects
    #confidence_threshold = navigatr_config['confidence_threshold']

        # processed_out = [None, None, None]
        # view_processed_frames(processed_out, False)

getOutput(obj_results, depth_results)

while True: 
     continue
     


# Load an image
