import time

import numpy as np
from src.sharable_data import thread_lock, depth_queue, depth_res_queue
import src.distance.get_depth as get_depth
from src.distance.get_depth_color_map import get_depth_color_map
import cv2

def run_dep_detect_model(frame=None, test_toggle: bool = False):

    print(f"Thread4: Running depth detection model...")

    frame_given = True  # Assume frame was given
    while True:
        #if thread_lock.acquire():
            if frame is None: # If frame was not given, get from queue and override frame_given assumption
                frame_given = False
                print(f"Thread4: Getting frame from depth_queue of queue size: {len(depth_queue)}")
                try:
                    thread_lock.acquire()
                    frame = depth_queue.get()
                    thread_lock.release()
                except: # If its still None, no more frames in queue
                    print(f"Thread4: Could not get frame from depth_queue")
                    thread_lock.release()
                    break # Stop if no more frames in queue
            if frame is None: # If its still None, no more frames in queue
                break  # Stop if no more frames in queue

            start_watch = time.time()
            # Run inference.
            prediction = get_depth.get_depth(frame)
            stop_watch = time.time()
            elapsed_time = stop_watch - start_watch
            depth = prediction["depth"]  # Depth in [m].
            # focallength_px = prediction["focallength_px"]  # Focal length in pixels.
            
            processed_frame = get_depth_color_map(depth)
            processed_frame.seek(0)
            processed_frame = np.frombuffer(processed_frame.read(), dtype=np.uint8)
            processed_frame = cv2.imdecode(processed_frame, cv2.IMREAD_COLOR)
            
            if test_toggle: # Log inference outputs
                pass

            result_packet = {
                "inference_time": elapsed_time,
                "depth_results": prediction,
                "processed_frame": processed_frame
            }
            print(f"Thread4: Detected dpeth map in {elapsed_time} seconds")
            thread_lock.acquire()
            depth_res_queue.put(result_packet)
            thread_lock.release()
            print(f"Thread4: depth_res_queue size: {len(depth_res_queue)}")

            frame = None

            # If frame was passed as an argument, break out of the loop (i.e. only run once)
            if frame_given:
                break
    print(f"Thread4: ...Finished running depth detection model.")
    return result_packet if frame_given else None
