#!/usr/bin/env python3
import json
import re
import os
import logging
import torch
from typing import List
import eval_utils as eval
import gradio as gr
import google.generativeai as genai
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from config import Config as cfg
from collections import defaultdict
import random


# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Gemini setup
# -----------------------
GOOGLE_API_KEY = cfg.gemini_api_key
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------
# MNIST loader
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist = datasets.MNIST(cfg.data_dir, train=False, download=False, transform=transform)

# --- Config ---
PATCH_SIZE = cfg.patch_size
DEVICE = cfg.device

# --- Globals for models ---
GLOB_GENERATOR = None
GLOB_CLASSIFIER = None
SAVE_DIR = "./gradio_tmp"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Model loading ---
def load_models(generator_path=None, classifier_path=None):
    global GLOB_GENERATOR, GLOB_CLASSIFIER
    from models.generator import ResidualGenerator
    from models.classifier import CNNClassifier
    GLOB_GENERATOR = ResidualGenerator(img_shape=(1,28,28)).to(DEVICE)
    GLOB_CLASSIFIER = CNNClassifier().to(DEVICE)
    if os.path.exists(generator_path):
        GLOB_GENERATOR.load_state_dict(torch.load(generator_path, map_location=DEVICE))
    if os.path.exists(classifier_path):
        GLOB_CLASSIFIER.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
    GLOB_GENERATOR.eval()
    GLOB_CLASSIFIER.eval()
    print("Models loaded successfully.")

# ------ Helpers ----
def run_transformation(x, label, target_digit:int, patches:List[int]):
    if GLOB_GENERATOR is None or GLOB_CLASSIFIER is None:
        return None, "Generator or classifier not loaded."
    x_vis = ((x + 1.0)/2.0).detach().cpu()
    print("patches: ", patches)
    eval.save_user_modification_example(
        x_vis=x_vis,
        simulated_patches=patches,
        generator=GLOB_GENERATOR,
        classifier=GLOB_CLASSIFIER,
        y_true=torch.tensor([label], device=DEVICE),
        y_target=torch.tensor([target_digit], device=DEVICE),
        device=DEVICE,
        save_dir=SAVE_DIR,
        patch_size=PATCH_SIZE
    )
    out_image = os.path.join(SAVE_DIR, "simulated_user_modification.png")
    return out_image, "Transformation complete."

digit_index = defaultdict(list)
for idx in range(len(mnist)):
    _, label = mnist[idx]
    digit_index[int(label)].append(idx)
def get_mnist_digit(d):
    # random sample from that digit's indices
    idx = random.choice(digit_index[d])
    x, y = mnist[idx]

    img = (x.squeeze(0) * 0.5 + 0.5).numpy() * 255
    img = Image.fromarray(img.astype(np.uint8), mode="L")
    img = img.resize((256, 256), Image.NEAREST)
    return x, img, y

# -----------------------
# JSON-action parser
# -----------------------
ACTION_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)

def extract_action(text):
    """
    Returns dict if valid action JSON exists, otherwise None.
    """

    m = ACTION_RE.search(text)
    if not m:
        return None

    candidate = m.group(1)
    try:
        parsed = json.loads(candidate)
    except:
        try:
            parsed = json.loads(candidate.replace("'", '"'))
        except:
            return None

    if parsed.get("action") == "show_digit":
        if isinstance(parsed.get("value"), int) and 0 <= parsed["value"] <= 9:
            return parsed
        
    if parsed.get("action") == "transform_digit":
        if "target" in parsed and "patches" in parsed:
            return parsed
    return None

# -----------------------
# Chat logic
# -----------------------
system_prompt = """
You are a helpful assistant for a MNIST counterfactual exploration tool.

RULES:
1. When the user requests a digit, you MUST output a JSON action at the END of your message.
2. The JSON must be wrapped inside an HTML comment so it is invisible:
   <!-- {"action":"show_digit","value":7} -->
3. Never show the JSON to the user. Speak naturally in the visible text.
4. Only one JSON action per message.
Example:
"Sure! Here is a sample digit."
<!-- {"action":"show_digit","value":3} -->

5. When performing a transformation, you MUST output a JSON with:
   - action: "transform_digit"
   - target: <digit 0-9>
   - patches: [list of patch integer indices each 0-15]
   If the user does not specify patches, you must allow all patches [0,1,2,...,15], but you should inform the user that they can specify allowed patches.
   Example:
   <!-- {"action":"transform_digit", "target":7, "patches":[0,1,5]} -->
6. Only do this when the user explicitly requests a transformation.

If no action is needed, just chat normally without JSON.
"""


history = []   # simple list of dicts
# Global variable to store the last selected MNIST digit
LAST_SELECTED_DIGIT = None
LAST_SELECTED_LABEL = None
LAST_SELECTED_DIGIT_PATCH_GRID = None

def chat(user_message, chat_history, img_display, transformed_display):

    """
    Handles user messages in the chatbot:
    - Show MNIST digit when user requests a digit
    - Transform last selected digit based on target_digit and allowed_patches
    """
    global LAST_SELECTED_DIGIT, LAST_SELECTED_LABEL, LAST_SELECTED_DIGIT_PATCH_GRID

    chat_history = chat_history or []         
    chat_history.append([user_message, None])

    # ---------- LLM call ----------
    prompt = system_prompt + "\n\nConversation:\n"
    for u, a in chat_history:
        prompt += f"user: {u}\n"
        if a is not None:
            prompt += f"assistant: {a}\n"
    prompt += "assistant:"

    response = model.generate_content(prompt).text

    # ---------- parse JSON action ----------
    action = extract_action(response)
    visible_text = re.sub(ACTION_RE, "", response).strip()
    chat_history[-1][1] = visible_text

    # ---------- handle actions ----------
    image = img_display  # default: previous image
    transformed_image = transformed_display

    if action:
        if action["action"] == "show_digit":
            # Pick a random MNIST digit
            d = int(action["value"])
            LAST_SELECTED_DIGIT, image, LAST_SELECTED_LABEL = get_mnist_digit(d)

            # Save patch grid visualization
            save_dir = os.path.join(cfg.save_dir, "eval_visuals/patch_grid.png")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            eval.visualize_patch_grid(LAST_SELECTED_DIGIT, cfg.patch_size, save_dir)
            LAST_SELECTED_DIGIT_PATCH_GRID = Image.open(save_dir)

        elif action["action"] == "transform_digit":
            if LAST_SELECTED_DIGIT is None:
                visible_text += "\nPlease first choose a digit to transform using the chatbot."
            else:
                target_digit = int(action["target"])
                allowed_patches = action["patches"]

                # Run counterfactual transformation
                transformed_path, msg = run_transformation(
                    LAST_SELECTED_DIGIT.unsqueeze(0).to(DEVICE),
                    label=LAST_SELECTED_LABEL,  # original label optional
                    target_digit=target_digit,
                    patches=allowed_patches
                )

                if transformed_path is not None:
                    transformed_img = Image.open(transformed_path)
                    transformed_image = transformed_img
                    visible_text += f"\n{msg}"
                else:
                    visible_text += f"\n{msg}"

    return "", chat_history, image, LAST_SELECTED_DIGIT_PATCH_GRID, transformed_image



def clear():
    history.clear()
    return [], None

# -----------------------
# Gradio UI
# -----------------------
with gr.Blocks() as demo:
    gr.Markdown("## Chatbot assistant for MNIST dataset exploration with Conditional Counterfactual GAN.")

    welcome = (
    "Welcome! You can explore MNIST digits and transform them into other digits. "
    "You can also control which regions (patches) are allowed to change. "
    "A patch reference image will be shown for guidance after you choose a digit to inspect.\n\n"
    "Which digit would you like to inspect first?"
    )
    chatbot = gr.Chatbot(value=[[None, welcome]])
    load_models(cfg.generator_path, cfg.classifier_path)

    with gr.Row():
        msg = gr.Textbox(show_label=False, placeholder="Type your message here...", container=False,scale=9)
        send = gr.Button("Send", variant="primary", scale=1)
    
    with gr.Row():
        image_output = gr.Image(label="Digit", height=256, width=256, scale=1)
        patch_grid = gr.Image(label="Patch grid", height=256, width=256, scale=1)
    
    with gr.Row():
        transformed_output = gr.Image(label="Transformed Digit")

    clear_btn = gr.Button("Clear")

    send.click(chat, [msg, chatbot, image_output, transformed_output],
           [msg, chatbot, image_output, patch_grid, transformed_output])

    msg.submit(chat, [msg, chatbot, image_output, transformed_output],
           [msg, chatbot, image_output, patch_grid, transformed_output])

    clear_btn.click(clear, None, [chatbot, image_output])

demo.launch(server_name="0.0.0.0")
