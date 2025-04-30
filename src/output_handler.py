from typing import List
import cv2
import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
import numpy as np
from src.sharable_data import thread_lock, output_stack
#from src.webcam import capture_one_frame, connect_to_webcam, run_camera_service, view_processed_frames
#from src.sharable_data import obj_res_queue, depth_res_queue, emot_res_queue


'''
# Load the Emotion Model, add this to the main script not here because we dont want it to load it over and over again 
# add this before the threads even start running, i.e. in the main thread at the start so it stays loaded
feature_extractor_layer = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2",
    trainable=False
)

# Ensure it's wrapped inside a model correctly
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),  # Explicit Input Layer
    tf.keras.layers.Lambda(lambda x: feature_extractor_layer(x)),  # Wrap Hub Layer in Lambda
    tf.keras.layers.Dense(7, activation='softmax')  # Change this based on your dataset
])
'''

# obj_output = result_packet_obj["detections"]
# depth_output = result_packet["processed_frame"]

def get_output(test_toggle:bool = False) -> List[str]:
    output = []
    thread_lock.acquire()
    # Loop until queue is empty
    while True:
        try:
            output.append(output_stack.get())
        except:
            break
    thread_lock.release()

    obj_output, depth_output = None, None
    if len(output) == 2:
        (obj_output, depth_output) = (output[0], output[1]) if isinstance(output[0][0][0], dict) else (output[1], output[0])
    else: return None

    print(f"Output Handler: obj_output: {obj_output} and depth_output: {depth_output}")

    boxes = []

    clockWidth = np.size(depth_output[0])
    # clockWidth = np.size(depth_output)
    clockQuad = clockWidth/4


    for s in(obj_output[0]):
    # for s in (obj_output):
        box_array=[]

        for i in  (int(s['box']['x1']), int(s['box']['x2'])):

            name = s['name']

            for j in (int(s['box']['y1']), int(s['box']['y2'])):
                x = []
                x.append(depth_output[j][i])
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

    strings = [0]*len(boxes)
    i = 0
    for s in (boxes):
        if test_toggle:
            print(( str(s["objame"]) + " "+ str(s["Direction"])+ " " + str(s["closest point"][0]) + " meters away\n" ))
        strings[i] = ( str(s["objame"]) + " "+ str(s["Direction"])+ " " + str(s["closest point"][0]) + " meters away" )
        i += 1

    return strings


def text_to_opencv(text: str):

    # Create a blank white image
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255

    # Choose font, scale, color, thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 0)  # black text
    thickness = 2

    # Put text on image
    cv2.putText(img, text, (10, 100), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    return img


def text_list_to_opencv(text_lines):
    # Create a blank white image
    line_height = 40
    img_height = max(200, line_height * len(text_lines) + 20)
    img = np.ones((img_height, 600, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 0, 0)
    thickness = 1
    y_start = 40  # Starting y position

    for i, line in enumerate(text_lines):
        y = y_start + i * line_height
        cv2.putText(img, line, (10, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    # cv2.imshow("Text Lines", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img

'''
# Load an image
image_path = '/home/naitikg2305/ENEE408Capstone/NaviGatr/src/EmotionDetec/efficientNet/20250312_225342.jpg'  # Update this with an actual image path
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Make a prediction
predictions = model.predict(image)
print("Predicted emotion scores:", predictions)
'''

