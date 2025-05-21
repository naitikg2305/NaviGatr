import requests
from io import BytesIO
import numpy as np
import cv2

API_KEY = "f7375dd876b2c26a2c19680ca1aadcb8136b06ed411c88a2aec95b8e92a58e6b"
API_URL = "https://3.231.131.94"
API_PORT = 8000

import os
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
API_CERT = (script_dir + "/cert.pem") if os.path.exists(script_dir + "/cert.pem") else "cert.pem"

def get_depth(img_ndarray):
    headers = {
        "accept": "application/octet-stream",
        "Authorization": f"Bearer {API_KEY}"
    }

    _, image = cv2.imencode(".jpg", img_ndarray)
    
    files = {
        "image": ("frame.jpg", image, "image/jpg")
    }

    response = requests.post(f"{API_URL}:{API_PORT}", headers=headers, files=files, verify=API_CERT)

    if response.status_code == 200:
        depth_data = BytesIO(response.content)
        depth_data.seek(0)

        return np.load(depth_data)
    
    else:
        print(f"Error: {response.status_code}")
        return None
    