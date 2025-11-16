#!/usr/bin/env python3
import json
import re
import os
import logging
import gradio as gr
import google.generativeai as genai
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from config import Config as cfg
from eval_utils import visualize_patch_grid
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
digit_index = defaultdict(list)
for idx in range(len(mnist)):
    _, label = mnist[idx]
    digit_index[int(label)].append(idx)

def get_mnist_digit(d):
    # random sample from that digit's indices
    idx = random.choice(digit_index[d])
    x, _ = mnist[idx]

    img = (x.squeeze(0) * 0.5 + 0.5).numpy() * 255
    img = Image.fromarray(img.astype(np.uint8), mode="L")
    img = img.resize((256, 256), Image.NEAREST)
    return x, img

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

If no action is needed, just chat normally without JSON.
"""


history = []   # simple list of dicts

def chat(user_message, chat_history, img_display):
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
    # Fill assistant side bubble
    chat_history[-1][1] = visible_text

    # ---------- handle action ----------
    image = None
    if action and action["action"] == "show_digit":
        d = int(action["value"])
        original_mnist_digit, image = get_mnist_digit(d)

        save_dir = os.path.join(cfg.save_dir, "eval_visuals/patch_grid.png")
        visualize_patch_grid(original_mnist_digit, cfg.patch_size, save_dir)
        patch_grid_img = Image.open(save_dir)


    return "", chat_history, image, patch_grid_img

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

    with gr.Row():
        msg = gr.Textbox(show_label=False, placeholder="Type your message here...", container=False,scale=9)
        send = gr.Button("Send", variant="primary", scale=1)
    
    with gr.Row():
        image_output = gr.Image(label="Digit", height=256, width=256, scale=1)
        patch_grid = gr.Image(label="Patch grid", height=256, width=256, scale=1)

    clear_btn = gr.Button("Clear")

    send.click(chat, [msg, chatbot, image_output], [msg, chatbot, image_output, patch_grid])
    msg.submit(chat, [msg, chatbot, image_output], [msg, chatbot, image_output, patch_grid])
    clear_btn.click(clear, None, [chatbot, image_output])

demo.launch(server_name="0.0.0.0")

