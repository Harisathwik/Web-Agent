import base64
import os
import requests

def encode_image(image_path):
    print("Encoding the image...")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def clear_screenshot_dir():
    if os.path.exists("screenshots"):
        for file in os.listdir("screenshots"):
            os.remove(os.path.join("screenshots", file))
        os.rmdir("screenshots")

    os.makedirs("screenshots")


def validate_url(url):
    """Validate the URL."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException as e:
        return False