import subprocess
from src.sharable_data import (frame_queue, thread_lock, output_stack_res,
                               obj_queue, depth_queue, emot_queue,
                               obj_res_queue, depth_res_queue, emot_res_queue,
                               colors, reset)


def capture_frame():
    result = subprocess.run(["libcamera-still", "-n", "-o", "-", "--encoding", "jpg"], 
                        stdout = subprocess.PIPE)

    if result.returncode != 0:
        print("Error from libcamera-still:", result.stderr.decode())
        raise RuntimeError("libcamera-still failed to run")

    if len(result.stdout) == 0:
        print("Warning: libcamera-still returned empty output")
        raise ValueError("No image data received")

    print(f"Received {len(result.stdout)} bytes from libcamera-still")
    
    image_bytes = result.stdout
    print(f"Type of image_bytes in capture.py {type(image_bytes)}.")
    
    
    return image_bytes