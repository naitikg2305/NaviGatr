import base64
import json
import os
import subprocess
import threading
import time
import cv2
import numpy as np
import torch
#from src.webcam import capture_one_frame, connect_to_webcam, run_camera_service, view_processed_frames
#from src.sharable_data import obj_res_queue, depth_res_queue, emot_res_queue

input_image_path = "input.jpg"
obj_output_json_path = "/home/naitikg2305/ENEE408April24/NaviGatr/obj_output.json" # Path to where output data will be stored
dep_output_json_path = "/home/naitikg2305/ENEE408April24/NaviGatr/dep_output.json" 

with open(obj_output_json_path, "r") as f:
        result_packet_obj = json.load(f)

print("SUCCESS: Received data back from run_obj.py")

    # Decode image and display
    
#processed_img_bytes = base64.b64decode(result_packet_obj["processed_frame"])
#processed_img_np = np.frombuffer(processed_img_bytes, dtype=np.uint8)
#processed_img = cv2.imdecode(processed_img_np, cv2.IMREAD_COLOR)


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
for s in(obj_results[0]):
    box_array=[]

    for i in  (int(s['box']['x1']), int(s['box']['x2'])):
        name = s['name']

        for j in (int(s['box']['y1']), int(s['box']['y2'])):
            x = []
            x.append(depth_results[j][i])
            box_array.append(x)
    boxes.append({"objame": name, 'closest point': min(box_array)})

print("----------------------------------------- \n \n \n")
print(boxes)



for s in(boxes):
     print(( "object " + str(s["objame"]) + " has closest point at " + str(s["closest point"])))
     print("\n")
# for i in range(0, 831):  
# for i in range(831, 1918):
    

#     for j in range(4, 1061):
#         x = []
#         x.append(depth_results[j][i])
#         box_array.append(x)

print(min(box_array))
print("SUCCESS: Received data back from run_depth.py")
   # Decode image and display
processed_img_bytes = base64.b64decode(result_packet["depth_image"])
processed_img_np = np.frombuffer(processed_img_bytes, dtype=np.uint8)
processed_img = cv2.imdecode(processed_img_np, cv2.IMREAD_COLOR)

#print(result_packet)
#print(processed_img_bytes)
#print(processed_img_np)
#print(processed_img)

result_packet["processed_frame"] = processed_img

    # push processed frame to queue



    # Find relevant objects and determine depth to objects
#confidence_threshold = navigatr_config['confidence_threshold']

    # processed_out = [None, None, None]
    # view_processed_frames(processed_out, False)
