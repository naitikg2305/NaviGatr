import time
from PIL import Image
import json
from src.sharable_data import thread_lock, depth_queue, depth_res_queue
import depth_pro




def run_dep_detect_model(frame=None, test_toggle: bool = False):

    print(f"Booting up depth detection model...")
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    time.sleep(15)  # Give time for camera service to generate its first frame
    print(f"Running depth detection model...")

    frame_given = True  # Assume frame was given
    while True:
        #if thread_lock.acquire():
            if frame is None: # If frame was not given, get from queue and override frame_given assumption
                frame_given = False
                print(f"Thread4: Getting frame from obj_queue of queue size: {len(depth_queue)}")
                frame = depth_queue.get()
            if frame is None: # If its still None, no more frames in queue
                break  # Stop if no more frames in queue
            # Load and preprocess an image.
            image, _, f_px = depth_pro.load_rgb(frame)
            image = transform(image)
            print(f"Thread4: Dimensions of input frame: {image.shape}")
            start_time = time.time()  # Timestamp
            # Run inference.
            prediction = model.infer(image, f_px=f_px)
            depth = prediction["depth"]  # Depth in [m].
            focallength_px = prediction["focallength_px"]  # Focal length in pixels.
            # thread_lock.release()
            
            if test_toggle: # Log inference outputs
                pass

            result_packet = {
                "timestamp": start_time,
                "focallength_px": focallength_px,
                "processed_frame": depth
            }
            print(f"Thread3: Detected objects in {time.time() - start_time} seconds")
            depth_res_queue.append(result_packet)
            print(f"Thread3: obj_res_queue size: {len(depth_res_queue)}")

            # If frame was passed as an argument, break out of the loop (i.e. only run once)
            if frame_given:
                return result_packet



