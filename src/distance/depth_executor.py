import requests
import cv2
from io import BytesIO
import numpy as np
import json
import os

dist_dir = os.path.dirname(__file__)
API_CERT_FILE = f"{dist_dir}/cert.pem"
API_CONFIG_FILE = f"{dist_dir}/api-config.json"

if not os.path.exists(API_CONFIG_FILE):
    raise FileNotFoundError("You must create a api-config.json file in the directory with the keys: API_KEY, API_URL, API_PORT.")

with open(f"{dist_dir}/api-config.json", "r") as api_config_file:
    api_config = json.load(api_config_file)

headers = {
        "accept": "application/octet-stream",
        "Authorization": f"Bearer {api_config['API_KEY']}"
    }
    
def get_depth(frame):
    _, image = cv2.imencode(".jpg", frame)

    file = {
        "image": ("frame.jpg", image, "image/jpg")
    }

    response = requests.post(f"{api_config['API_URL']}:{api_config['API_PORT']}", headers=headers, files=file, verify=API_CERT_FILE)

    if response.status_code == 200:
        depth_data = BytesIO(response.content)
        depth_data.seek(0)

        return np.load(depth_data)
    
    elif response.status_code == 401:
        raise PermissionError("You API key is not valid - please check the server for the correct key.")
    
    else:
        print(f"{response.status_code}: Unknown API error - something is wrong with connection to server")
        return None