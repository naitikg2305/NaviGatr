import time
import numpy as np
import cv2
import threading
from typing import List
# from picamera2 import Picamera2 # Only available on Linux-based systems
from src.sharable_data import (frame_queue, thread_lock, output_stack_res,
                               obj_queue, depth_queue, emot_queue,
                               obj_res_queue, depth_res_queue, emot_res_queue,
                               colors, reset)

class FrameGrabber:
    def __init__(self, cam_url):
        self.cap = cv2.VideoCapture(cam_url, cv2.CAP_FFMPEG)
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame

    def read(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.cap.release()



def run_camera_service(camera_type: str, ip_cam_addr: str=None, test_toggle: bool = False):
    print(f"Running camera service...\nCamera type: {camera_type}, IP address: {ip_cam_addr}, Test mode: {test_toggle}")
    if camera_type == "webcam" or camera_type == "webcam_single":
        capture: cv2.VideoCapture
        raw_out: cv2.VideoWriter
        processed_out: cv2.VideoWriter
        capture, raw_out, processed_out = connect_to_webcam(ip_cam_addr=ip_cam_addr, test_toggle=test_toggle)

        # Capture video frames from active webcam
        if camera_type == "webcam_multi":
            capturing_frames_thread = threading.Thread(target=capture_frame, args=(capture, raw_out, test_toggle), daemon=True)
        elif camera_type == "webcam_single":
            capturing_frames_thread = threading.Thread(target=capture_frame, args=(camera_type, capture, raw_out, test_toggle), daemon=True)
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")
        capturing_frames_thread.start()

        # Display video frames processed by ML models
        viewing_obj_frames_thread = threading.Thread(target=view_processed_frames, args=(processed_out, test_toggle), daemon=True)
        viewing_obj_frames_thread.start()
        # Running capture_frames and view_processed_frames in parallel

        # Shutdown webcam when one of the threads finishes
        while capturing_frames_thread.is_alive() and viewing_obj_frames_thread.is_alive():
            time.sleep(5)
        shutdown_webcam(capture, raw_out, processed_out)

    elif camera_type == "picam":
        pass

    else:
        raise ValueError(f"Unknown camera type: {camera_type}")

''' Uncomment on Linux-based system
def connect_to_rpi_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
    picam2.start()

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR (OpenCV format)

        if not frame_queue.full():
            frame_queue.put(frame)

        if not result_queue.empty():
            result_packet = result_queue.get()

            # Access the processed frame for visualization
            processed_frame = result_packet["processed_frame"]
            detections = result_packet["detections"]
            timestamp = result_packet["timestamp"]

            print(f"Frame Timestamp: {timestamp}, Detections: {len(detections)} objects found.")

            cv2.imshow("Raspberry Pi Camera", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    picam2.stop()
    frame_queue.put(None)  # Stop the inference thread
    cv2.destroyAllWindows()


def test_picam():
    time_now = time.ctime()

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
    picam2.start()

    frame = picam2.capture_array()
    cv2.imwrite(f"picam_img_{time_now}.jpg", frame)
    picam2.stop()

    print(f"picam test image as picam_img_{time_now}.jpg")
#'''

def connect_to_webcam(ip_cam_addr, test_toggle: bool = False):
    """
    Connects to the IP camera and processes frames in a separate thread.
    If test_toggle is True, saves raw and processed feed to local files.
    
    Args:
        test_toggle (bool): If True, saves raw and processed feed to local files.
        direct_model_call (bool): If True, calls the model directly from the main thread.

    Returns:
        None
    """
    print(f"Connecting to IP camera...")
    cam_url = ip_cam_addr
    frame_grabber = FrameGrabber(cam_url)  # Connect to the camera and assign to device object 'cap'

    processed_out_obj = None
    processed_out_dep = None
    processed_out_emo = None
    raw_out = None

    if test_toggle:
        # Wait until we get a valid frame to determine dimensions
        import time
        frame = None
        timeout = time.time() + 5  # wait max 5 seconds
        while frame is None and time.time() < timeout:
            frame = frame_grabber.read()

        if frame is None:
            raise RuntimeError("Failed to get a frame from the IP camera for video writer setup.")

        frame_height, frame_width = frame.shape[:2]
        fps = 15  # Set a fixed FPS that matches expected camera output

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        raw_out = cv2.VideoWriter("raw_feed.avi", fourcc, fps, (frame_width, frame_height))
        processed_out_obj = cv2.VideoWriter("processed_feed_obj.avi", fourcc, fps, (frame_width, frame_height))
        processed_out_dep = cv2.VideoWriter("processed_feed_dep.avi", fourcc, fps, (frame_width, frame_height))
        processed_out_emo = cv2.VideoWriter("processed_feed_emo.avi", fourcc, fps, (frame_width, frame_height))

    return frame_grabber, raw_out, [processed_out_obj, processed_out_dep, processed_out_emo]

def capture_frame(camera_type: str, capture: cv2.VideoCapture, raw_out:cv2.VideoWriter=None, test_toggle: bool = False):

    if camera_type == "webcam_single" or camera_type == "webcam_multi":
        print(f"ThreadM: Capturing one frame...") if test_toggle else None
        if capture.isOpened():

            frame = capture.read()
            
            if frame is None:
                print("\nThreadM: Frame is None\n") if test_toggle else None
                return

            if test_toggle:
                raw_out.write(frame)

            if thread_lock.acquire():
                try:
                    frame_queue.put(frame)
                    obj_queue.put(frame)
                    depth_queue.put(frame)
                    emot_queue.put(frame)
                    print(f"ThreadM: Single frame added to queue. The frame_queue is now size: {len(frame_queue)}") if test_toggle else None
                except:
                    print("ThreadM: Frame queue is full. Skipping frame...")  if test_toggle else None
                thread_lock.release()
    return frame


def view_processed_frames(processed_out, test_toggle: bool):
    processed_out:List[cv2.VideoWriter]

    print(f"{colors['steel_gray']}Thread2: Viewing processed frames... {reset}") if test_toggle else None
    blank_image = np.zeros((400, 400, 3), dtype=np.uint8)  # Create a blank image as placeholders
    margin = np.zeros((400, 20, 3), dtype=np.uint8)
    # Create a window with three slots
    cv2.namedWindow("Models' Results", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Models' Results", 1660, 400) # 400*4 + 20*3
    # Display the blank image in each slot
    cv2.imshow("Models' Results", np.hstack((blank_image, margin, blank_image, margin, blank_image, margin, blank_image)))
    # Move the window to the left slot
    cv2.moveWindow("Models' Results", 0, 0)
    
    br = False
    obj_slot = blank_image
    dep_slot = blank_image
    cap_slot = blank_image
    out_slot = blank_image
    frame_count = 0
    while br == False:
        obj_bool = True
        dep_bool = True
        cap_bool = True
        out_bool = True
        time.sleep(0.5)
        thread_lock.acquire()  # Acquire the lock for sharable_data.py

        # TRY: Get frame in each result queue EXCEPT: Queue is empty
        try: 
            obj_result_packet = obj_res_queue.get()
        except:
            obj_bool = False
        try:
            dep_result_packet = depth_res_queue.get()
        except:
            dep_bool = False
        try:
            cap_result_packet = frame_queue.get()
        except:
            cap_bool = False
        try:
            out_result_packet = output_stack_res.get()
        except:
            out_bool = False
        thread_lock.release()

        if obj_bool: # If frame was on object result queue
            # Access the processed frame for visualization
            processed_frame = obj_result_packet["processed_frame"]
            detections = obj_result_packet["detections"]
            inference_time = obj_result_packet["inference_time"]

            if processed_frame is not None:
                print(f"{colors['steel_gray']}Thread2: Processed frame is not none{reset}") if test_toggle else None
                if test_toggle:
                    processed_out[0].write(processed_frame.copy())  # Write processed frame to file
                    frame_count += 1  # Increment frame count
                    print(f"{colors['steel_gray']}Thread2: [Saved Frame #{frame_count}] Inference Time: {inference_time}{reset}")
                obj_slot = processed_frame # Update object display image to most recent inference


        if dep_bool: # If frame was on depth result queue
            # Access the processed frame for visualization
            processed_frame = dep_result_packet["processed_frame"]
            inference_time = dep_result_packet["inference_time"]

            if processed_frame is not None:
                print(f"{colors['steel_gray']}Thread2: Processed frame is not none{reset}") if test_toggle else None
                if test_toggle:
                    processed_out[1].write(processed_frame.copy())  # Write processed frame to file
                    frame_count += 1  # Increment frame count
                    print(f"{colors['steel_gray']}Thread2: [Saved Frame #{frame_count}] Inference Time: {inference_time}{reset}")
                dep_slot = processed_frame # Update depth display image to most recent inference

        if cap_bool: # If new image has been captured
            cap_slot = cap_result_packet # Update cap display image to most recent capture

        if out_bool:
            out_slot = out_result_packet

        # Display frames with most recent results
        obj_slot = cv2.resize(obj_slot, (400, 400), interpolation=cv2.INTER_AREA)
        dep_slot = cv2.resize(dep_slot, (400, 400), interpolation=cv2.INTER_AREA)
        cap_slot = cv2.resize(cap_slot, (400, 400), interpolation=cv2.INTER_AREA)
        out_slot = cv2.resize(out_slot, (400, 400), interpolation=cv2.INTER_AREA)
        cv2.imshow("Models' Results", np.hstack((cap_slot, margin, obj_slot, margin, dep_slot, margin, out_slot)))

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            br = True
    


def shutdown_webcam(capture: cv2.VideoCapture, raw_out: cv2.VideoWriter, processed_out: cv2.VideoWriter, test_toggle: bool = False):
    '''
    Shuts down the webcam and releases resources.
    '''
    print(f"Thread M: Shutting down webcam...")
    capture.release()
    if test_toggle:
        raw_out.release()
        processed_out.release()
        print(f"Thread M: ...Saved videos were released")
    frame_queue.put(None)  # Stop the inference thread
    obj_queue.put(None)
    depth_queue.put(None)
    emot_queue.put(None)
    cv2.destroyAllWindows()
    time.sleep(1)


def flush_camera_buffer(cap: cv2.VideoCapture, max_flush_time=0.3):
    start = time.time()
    while time.time() - start < max_flush_time:
        cap.grab()
