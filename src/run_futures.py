import concurrent.futures as cf

if __name__ == "__main__":
    from distance.depth_executor import get_depth
    from objectDetection.object_executor import init_object_executor, get_objects
    from EmotionDetec.emotion_executor import init_emotion_executor, get_emotion
    from distance.api_liveness import liveness_check
    import subprocess
    import cv2
    import numpy as np
    import num2words
    from espeakng import ESpeakNG

    # Ensure that the depth API server is up and running!
    liveness_check()

    # Initialize text to speech
    voice = ESpeakNG()

    import multiprocessing
    multiprocessing.set_start_method("spawn")

    depth_executor = cf.ProcessPoolExecutor(max_workers=1)
    obj_executor = cf.ProcessPoolExecutor(max_workers=1, initializer=init_object_executor)
    emot_executor = cf.ProcessPoolExecutor(max_workers=1, initializer=init_emotion_executor)

    futures = {"depth": None, "object": None, "emotion": None}

    def capture_frame():
        result = subprocess.run([
            "libcamera-still", "-n", "-t", "100", "-o", "-", "--encoding", "jpg"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0 or len(result.stdout) == 0:
            raise OSError("Camera capture failed...")

        img_array = np.frombuffer(result.stdout, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return frame

    try:
        while True:
            frame = capture_frame()

            futures["depth"] = depth_executor.submit(get_depth, frame)
            futures["object"] = obj_executor.submit(get_objects, frame)

            # Wait for the depth and object models to finish running
            cf.wait(futures.values())

            # Future results
            # The second depth is indexing into the numpy array as the estimated/EXIF focal length
            # is also returned
            depth_map = futures["depth"].result()["depth"]
            objects = futures["object"].result()

            clockWidth = np.size(depth_map[0])

            # Split the return depth map into quadrants
            clockQuad = clockWidth / 4
            
            output = []

            for obj in objects:
                # Get the central x value of the bounding box
                center_x = (obj["box"][0] + obj["box"][2])/2

                # Divide the field of view into 8 slices where each "hour" gets two slices except for the extreme values, 10 and 2
                clock_labels = ["10 o'clock", "11 o'clock", "11 o'clock", "12 o'clock", "12 o'clock", "1 o'clock", "1 o'clock", "2 o'clock",]

                # Find the location within the eight octiles of vision
                octile = min(int(8 * (center_x / clockWidth)), 7)
                direction = clock_labels[octile]

                # get object distance
                distance = np.min(depth_map[obj["box"][0]:obj["box"][2], obj["box"][1]:obj["box"][3]])

                # Default if there is no person
                emotion = None
                confidence = 0

                if obj["label"] == "person" and distance < 1:
                    futures["emotion"] = emot_executor.submit(get_emotion, frame[obj["box"][1]:obj["box"][3], obj["box"][0]:obj["box"][2]])

                    # Wait for emotion model to complete
                    cf.wait(futures.values())

                    emotion, confidence = futures["emotion"].result()

                output.append({
                        "label": obj["label"],
                        "closest point": distance,
                        "Direction": direction,
                        "Emotion": {emotion},
                        "Confidence": {confidence * 100}
                    })
                
            for obj in output:
                if obj["Emotion"] is not None:
                    print(f"{obj['label']} {obj['Emotion']} {obj['Direction']} {obj['closest point'][0]:.2f} meters away")
                    voice.say(f"{obj['label']} {obj['Emotion']} {obj['Direction']} {num2words(round(obj['closest point'][0], 2))} meters away")
                else:
                    print(f"{obj['label']} {obj['Direction']} {obj['closest point'][0]:.2f} meters away")
                    voice.say(f"{obj['label']} {obj['Direction']} {num2words(round(obj['closest point'][0], 2))} meters away")



    except KeyboardInterrupt:
        print("Killing processes...")
        depth_executor.shutdown()
        obj_executor.shutdown()
        emot_executor.shutdown()
        print("NaviGatr has terminated")
