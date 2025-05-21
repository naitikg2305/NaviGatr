import requests
import os
import json

dist_dir = os.path.dirname(__file__)
API_CERT_FILE = f"{dist_dir}/cert.pem"
API_CONFIG_FILE = f"{dist_dir}/api-config.json"

if not os.path.exists(API_CONFIG_FILE):
    raise FileNotFoundError("You must create a api-config.json file in the directory with the keys: API_KEY, API_URL, API_PORT.")

with open(f"{dist_dir}/api-config.json", "r") as api_config_file:
    api_config = json.load(api_config_file)

headers = {
        "accept": "application/octet-stream",
        "Authorization": f"Bearer {api_config["API_KEY"]}"
    }

# Does connectivity test with server
def liveness_check():
    # Does the cert file exist
    if not os.path.exists(API_CERT_FILE):
        raise FileNotFoundError(f"""{API_CERT_FILE} does not exist in this directory.
              Download the certificate from your server and place it in the distance directory.
              Otherwise, remove https protections or use a properly signed cert for your domain name.""")
    
    # Liveness check
    try:
        global headers
        response = requests.get(f"{api_config["API_URL"]}:{api_config["API_PORT"]}", headers=headers, verify=API_CERT_FILE, timeout=5)
        
        if response.status_code == 200:
            print("Server up and running, API key correct, and SSL cert correct!")
        elif response.status_code == 401:
            raise PermissionError("You API key is not valid - please check the server for the correct key.")
        else:
            raise Exception(f"{response.status_code}: Unknown API error - something is wrong with connection to server")
    except requests.exceptions.ConnectTimeout as e:
        raise requests.exceptions.ConnectTimeout("It appears the servers is not on or is misconfigured. Liveness check failed.")
    except requests.exceptions.SSLError as e:
        raise requests.exceptions.SSLError("""It appears your SSL cert is invalid - check with the server for the proper cert,
              get rid of SSL altogether, or get a public domain name cert.""")