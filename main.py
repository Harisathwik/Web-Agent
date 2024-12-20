from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import (FunctionMessage, SystemMessage, BaseMessage)
from langchain.tools.render import format_tool_to_openai_function
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from playwright.sync_api import sync_playwright, Page
import os
import json
import operator
import ast
from datetime import datetime
from utils import encode_image, clear_screenshot_dir, validate_url
import uuid
from qwen_vl_utils import process_vision_info
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv

load_dotenv()
# Set API KEYS
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# System prompt
SYSTEM_PROMPT = f'''
You are a web assistant who navigates on the browser just like humans. Always invoke the tools to search for accurate and updated information online.
Information and data is available under the variable [state].
REMEMBER THE RULES:
- Never provide information based on your knowledge cutoff date or earlier than {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
- Use the tools any number of times to get the desired information.
- Always take screenshot after every tool and do perform image analysis tool action and answer based on this.
- Never call analyze_image tool or fetch_coordinates tool without a screenshot.
- Never call click_coordinates tool without coordinates.
- Never return an empty response.'''


from utils import AgentState
# Encapsulate the state in a class
# class AgentState(TypedDict):
#     """Encapsulates the agent's state for message passing and browser interactions."""
#     page: Page  # Playwright page object
#     messages: Annotated[Sequence[BaseMessage], operator.add]  # List of AnyMessage (HumanMessage, SystemMessage, etc.)
#     query: str  # Query for the current task
#     screenshot_path: []  # Path to the screenshot image
#     image_analysis: []  # Analysis of the image
#     coordinates: []  # Coordinates of the location on the screenshot


@tool
def go_back(state):
    """Navigate back to the previous page."""
    print("Navigating back to the previous page...")
    state["page"].go_back()

    function_message = FunctionMessage(
        name="go_back",
        content=f'Navigating back to the previous page.',
    )
    state["messages"].append(function_message)

    screenshot(state)


def summarize_query(query, prompt='Shorten this sentence into single line to search in the browser for getting best results. JUST THE ANSWER AND NO EXTRA INFORMATION.'):
    refined_query = llm.invoke([SystemMessage(
        content=prompt + query)])
    return str(refined_query.content)


def generate_query(state):

    response = llm.invoke([SystemMessage(content=f'Here are the web page page contents: {state["image_analysis"][-1]} and my conversation:{state["messages"]}. Tell me what to be clicked to get the desired information? NOTE: Only single line answer is expected.')])
    return response


@tool
def fetch_coordinates(state):
    """Fetch coordinates of the first search result from the screenshot took earlier."""
    min_pixels = 256 * 28 * 28
    max_pixels = 1344 * 28 * 28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    _SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."

    response = generate_query(state)
    state['query'] = response.content

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SYSTEM},
                {"type": "image", "image": state["screenshot_path"][-1], "min_pixels": min_pixels,
                 "max_pixels": max_pixels},
                {"type": "text", "text": state["query"]},
            ],
        }
    ]

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "showlab/ShowUI-2B", torch_dtype=torch.bfloat16, device_map="auto"
    )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    click_xy = ast.literal_eval(output_text)
    state["coordinates"] = [click_xy[0]*image_inputs[0].width, click_xy[1]*image_inputs[0].height]  # Placeholder coordinates
    print("State after fetching coordinates:", state)

    tool_name = state["messages"][-1].additional_kwargs['tool_calls'][0]['function']['name']
    function_message = FunctionMessage(
        name=tool_name,
        content=f'Fetched coordinates {state["coordinates"]} for the first search result.',
    )
    state["messages"].append(function_message)

    click_coordinates(state)
    screenshot(state)
    return state

# Define action functions
@tool
def browser_type_text(state):
    """Type text in the browser's address bar."""
    last_message = state["messages"][-1]
    arguments = last_message.additional_kwargs["tool_calls"][0]["function"]["arguments"]
    tool_name = last_message.additional_kwargs['tool_calls'][0]['function']['name']
    arguments = eval(arguments)

    if 'url' not in arguments['state']:
        state["query"] = summarize_query(state["query"])
        query = str(state["query"]).replace('"', '')
        # query = state["query"]

        state["page"].keyboard.type(query)
        state["page"].keyboard.press("Enter")
        print(f'Typed: {query}.')

        function_message = FunctionMessage(
            name=tool_name,
            content=f'Typed {query} in the address bar.',
        )
        state["messages"].append(function_message)
    else:
        go_to_url(state)

    return state


def screenshot(state):
    """Take a screenshot of the current page to use analyze_image tool later."""
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")

    name = uuid.uuid4().hex
    print("Taking a screenshot of the current page...")
    state["page"].screenshot(path=f'screenshots/{name}.png')
    state["screenshot_path"].append(f'screenshots/{name}.png')
    print(f'Screenshot saved at screenshots/{name}.png')
    state["screenshot_path"].append(f'screenshots/{name}.png')

    return state

@tool
def analyze_image(state):
    """Analyze the web page screenshot image to understand contents and text from it."""
    # llm = ChatOpenAI(
    #     model="gemma2-9b-it",
    #     temperature=0,
    #     base_url="https://api.groq.com/openai/v1",
    # )
    # base64_image = encode_image(state["screenshot_path"][-1])
    print("Analyzing the contents of the image...")
    # output_message = llm.invoke(
    #     [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": "Fetch all the text from the image and analyse the contents."},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/png;base64,{base64_image}",
    #                     },
    #                 },
    #             ],
    #         }
    #     ]
    # )
    output_message = state['page'].evaluate("document.body.innerText")
    output_message = summarize_query(output_message, prompt="You are summariser who summarises given query:")
    # state["image_analysis"].append(output_message.content)
    state["image_analysis"].append(output_message)

    print("Image analysis:", state["image_analysis"][-1])
    tool_name = state["messages"][-1].additional_kwargs['tool_calls'][0]['function']['name']
    function_message = FunctionMessage(
        name=tool_name,
        content=json.dumps(state["messages"][-1].content),
    )
    state["messages"].append(function_message)

    screenshot(state)
    return state


def click_coordinates(state):
    """Click on the specified coordinates on the screenshot."""
    state["page"].mouse.click(state["coordinates"][0], state["coordinates"][1])
    print(f'Clicked on the coordinates {state["coordinates"]}.')

    function_message = FunctionMessage(
        name="click_coordinates",
        content=f'Clicked on the coordinates {state["coordinates"]}.',
    )
    state["messages"].append(function_message)

    return state

def go_to_url(state):
    """Navigate to the specified URL."""
    last_message = state["messages"][-1]
    arguments = last_message.additional_kwargs["tool_calls"][0]["function"]["arguments"]
    tool_name = last_message.additional_kwargs['tool_calls'][0]['function']['name']
    arguments = eval(arguments)

    valid_url = validate_url(arguments["state"]["url"])

    if valid_url:
        state["page"].goto(arguments["state"]["url"])
        print(f'Navigated to {arguments["state"]["url"]}.')

        function_message = FunctionMessage(
            name=tool_name,
            content=f'Navigated to {arguments["state"]["url"]}.',
        )
        state["messages"].append(function_message)
    else:
        print(f'Invalid URL: {arguments["state"]["url"]}')

        function_message = FunctionMessage(
            name=tool_name,
            content=f'Invalid URL: {arguments["state"]["url"]}.',
        )
        state["messages"].append(function_message)

    return state


# Function mapping
action_map = {
    "browser_type_text": browser_type_text,
    "analyze_image": analyze_image,
    "go_back": go_back,
    "fetch_coordinates": fetch_coordinates,
}

# Function to determine the next action
def call_model(state: AgentState):
    """Run the agent to process the task."""
    if state["image_analysis"]:
        image_analysis = state["image_analysis"][-1]
    else:
        image_analysis = []
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"] + [SystemMessage(content=f'Try answering the query now from the image analysis {image_analysis}. If you cannot, What is the immediate next step you would like to perform using the tools??')]
    response = model.invoke(messages)
    print("Model response:", response)
    state["messages"].append(response)


# Function to call the appropriate action
def call_action(state):
    """Call the appropriate action based on the model's response."""
    print("Calling the action...")
    messages = state["messages"]
    last_message = messages[-1]

    # Parse action name and arguments from the model's response
    tool_call = last_message.additional_kwargs['tool_calls'][0]
    tool_name = tool_call['function']['name']
    tool_args = tool_call['function']['arguments']

    # Deserialize tool_args if it's a string
    if isinstance(tool_args, str):
        tool_args = json.loads(tool_args)  # Safely parse JSON string to dict

    # Ensure `state` is included in the tool arguments
    tool_args['state'] = state

    # Execute the corresponding action
    if tool_name in action_map:
        state = action_map[tool_name](tool_args)  # Pass the tool_args dict
    else:
        raise ValueError(f'Unknown tool: {tool_name}')


# Function to decide if the process should continue
def should_continue(state: AgentState):
    """Check if the agent should continue processing the task."""
    print("Checking if the agent should continue...")
    last_message = state["messages"][-1]
    if "tool_calls" not in last_message.additional_kwargs:
        state["page"].close()
        return "end"
    else:
        return "continue"


# ============================== MODEL ==============================
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,  # Deterministic
)


tools = [browser_type_text, analyze_image, go_back, fetch_coordinates]
functions = [format_tool_to_openai_function(tool) for tool in tools]

model = llm.bind_functions(functions)

# ============================== BROWSER ==============================
browser = sync_playwright().start().chromium.launch(headless=False)
page = browser.new_page()
page.goto("https://google.com")

# Initialize the agent state with messages
state = AgentState(
    page=page,
    messages=[SystemMessage(content=SYSTEM_PROMPT)],  # Initialize with the system prompt
    query="",
    screenshot_path=[],
    image_analysis=[],
    coordinates=[],
)

# ============================== GRAPH ==============================
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_action)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    }
)

workflow.add_edge("action", "agent")

app = workflow.compile()

# ============================== MAIN LOOP ==============================
from langchain_core.messages import HumanMessage

input_content = "Latest video from fittuber channel?" # input("Enter your message: ")

clear_screenshot_dir()

# if input_content in ["exit", "quit", "q"]:
#     break
input_message = {"page": page, "messages": [HumanMessage(content=input_content)], "query": f'{input_content}', "screenshot_path": [], "image_analysis": [], "coordinates": []}
app.invoke(input_message)


"""
TO-DO:
- Add padding to click.
- Take query for whose coordinates are to be fetched.
- Implement SoM (Set of Marks) as tool.
- Implement the agent's memory.
- Implement the agent's learning capabilities.
- Implement the agent's reasoning capabilities.
- Add fetch_coordinates tool to get the coordinates of a location.
- Human in loop for taking inputs for filling forms.
"""