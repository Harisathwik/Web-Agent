import base64
import os
import requests
from typing_extensions import TypedDict
from typing import Sequence, Annotated
from langchain_core.messages import BaseMessage
import operator
from playwright.sync_api import Page


# Encapsulate the state in a class
class AgentState(TypedDict):
    """Encapsulates the agent's state for message passing and browser interactions."""
    page: Page  # Playwright page object
    messages: Annotated[Sequence[BaseMessage], operator.add]  # List of AnyMessage (HumanMessage, SystemMessage, etc.)
    query: str  # Query for the current task
    screenshot_path: []  # Path to the screenshot image
    image_analysis: []  # Analysis of the image
    coordinates: []  # Coordinates of the location on the screenshot


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
