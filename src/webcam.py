import time
import cv2
import threading
# from picamera2 import Picamera2 # Only available on Linux-based systems
from src.sharable_data import (frame_queue, thread_lock,
                               obj_queue, depth_queue, emot_queue,
                               obj_res_queue, depth_res_queue, emot_res_queue)

def run_camera_service(camera_type: str, ip_cam_addr: str=None, test_toggle: bool = False):
    print(f"Running camera service...\nCamera type: {camera_type}, IP address: {ip_cam_addr}, Test mode: {test_toggle}")
    if camera_type == "webcam":
        capture: cv2.VideoCapture
        raw_out: cv2.VideoWriter
        processed_out: cv2.VideoWriter
        capture, raw_out, processed_out = connect_to_webcam(ip_cam_addr=ip_cam_addr, test_toggle=test_toggle)

        # Capture video frames from active webcam
        capturing_frames_thread = threading.Thread(target=capture_frames, args=(capture, raw_out, test_toggle), daemon=True)
        capturing_frames_thread.start()
        # Display video frames processed by ML models
        viewing_obj_frames_thread = threading.Thread(target=view_processed_frames, args=("obj_det", processed_out, test_toggle), daemon=True)
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
    capture = cv2.VideoCapture(cam_url, apiPreference=cv2.CAP_FFMPEG)  # Connect to the camera and assign to device object 'cap'

    processed_out_obj = None
    processed_out_dep = None
    processed_out_emo = None
    raw_out = None

    if test_toggle:  # Run to save raw and processed feed to local files
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = int(fps) if fps and fps > 0 else 30  # fallback if FPS not detected
        fps = 15  # Set a reasonable FPS manually (make sure this matches your camera feed FPS)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        raw_out = cv2.VideoWriter("raw_feed.avi", fourcc, fps, (frame_width, frame_height))
        processed_out_obj = cv2.VideoWriter("processed_feed_obj.avi", fourcc, fps, (frame_width, frame_height))
        processed_out_dep = cv2.VideoWriter("processed_feed_dep.avi", fourcc, fps, (frame_width, frame_height))
        processed_out_emo = cv2.VideoWriter("processed_feed_emo.avi", fourcc, fps, (frame_width, frame_height))

    return capture, raw_out, [processed_out_obj, processed_out_dep, processed_out_emo]

def capture_frames(capture: cv2.VideoCapture, raw_out:cv2.VideoWriter=None, test_toggle: bool = False):
    print(f"Capturing frames...")

    # Until webcam has been closed
    while capture.isOpened():
        #if thread_lock.acquire():
        time.sleep(0.04)
        ret, frame = capture.read()
        if not ret:
            break

        if frame is None:
            print("\nThread1: Frame is None\n")
        
        if test_toggle:
            raw_out.write(frame)

        try:
            frame_queue.put(frame)
            obj_queue.put(frame)
            depth_queue.put(frame)
            emot_queue.put(frame)
        except:
            print("Thread1: Frame queue is full. Skipping frame...") 

            #thread_lock.release()

def view_processed_frames(model:str, processed_out, test_toggle: bool):
    print(f"Viewing processed frames...")
    br = False
    res_queue = None
    # Switch to the desired model
    if model == "obj_det":
        res_queue = obj_res_queue
        processed_out = processed_out[0]
    elif model == "depth_det":
        res_queue = depth_res_queue
        processed_out = processed_out[1]
    elif model == "emot_det":
        res_queue = emot_res_queue
        processed_out = processed_out[2]

    processed_out:cv2.VideoWriter

    frame_count = 0
    while br == False:
        time.sleep(0.02)
        try:
            result_packet = res_queue.pop()
        except:
            continue

        # Access the processed frame for visualization
        processed_frame = result_packet["processed_frame"]
        detections = result_packet["detections"]
        timestamp = result_packet["timestamp"]

        if processed_frame is not None:
            print(f"\nThread2: Processed frame is not none")
            if test_toggle:
                processed_out.write(processed_frame.copy())  # Write processed frame to file
                frame_count += 1  # Increment frame count
                print(f"Thread2: [Saved Frame #{frame_count}] Timestamp: {timestamp}")

        cv2.imshow("Phone IP Camera", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            br = True


def shutdown_webcam(capture: cv2.VideoCapture, raw_out: cv2.VideoWriter, processed_out: cv2.VideoWriter, test_toggle: bool = False):
    '''
    Shuts down the webcam and releases resources.
    '''
    print(f"Shutting down webcam...")
    capture.release()
    if test_toggle:
        raw_out.release()
        processed_out.release()
        print(f"...Saved videos were released")
    frame_queue.put(None)  # Stop the inference thread
    obj_queue.put(None)
    depth_queue.put(None)
    emot_queue.put(None)
    cv2.destroyAllWindows()
    time.sleep(1)

def capture_one_frame(capture: cv2.VideoCapture, raw_out:cv2.VideoWriter=None, test_toggle: bool = False):

    if capture.isOpened():
        #if thread_lock.acquire():
        time.sleep(0.04)
        ret, frame = capture.read()
        if not ret:
            return

        if frame is None:
            print("\nThread1: Frame is None\n")
        
        if test_toggle:
            raw_out.write(frame)

        try:
            frame_queue.put(frame)
            obj_queue.put(frame)
            depth_queue.put(frame)
            emot_queue.put(frame)
        except:
            print("Thread1: Frame queue is full. Skipping frame...") 
            #thread_lock.release()
