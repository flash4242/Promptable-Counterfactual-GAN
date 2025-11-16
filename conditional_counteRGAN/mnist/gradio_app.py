#!/usr/bin/env python3
import os
import re
import json
import random
import torch
import torchvision
from torchvision import transforms, datasets
import gradio as gr
from typing import List, Optional
import eval_utils as eval
from config import Config as cfg

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure API key
try:
    GENAI_API_KEY = cfg.gemini_api_key
    if not GENAI_API_KEY:
        raise ValueError("Missing GENAI_API_KEY environment variable")
    genai.configure(api_key=GENAI_API_KEY)
except Exception as e:
    logger.error(f"API Configuration Error: {e}")
    raise

# --- Config ---
PATCH_SIZE = cfg.patch_size
DEVICE = cfg.device

# MNIST dataset loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_test = datasets.MNIST(cfg.data_dir, train=False, transform=transform, download=False)

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


class GeminiChatbot:
    def __init__(self):
        self.conversation_history = []
        self.setup_model()
        
    def setup_model(self):
        try:
            # Template for our chat prompt
            template = f"""
            You are a friendly and knowledgeable assistant that guides users through generating counterfactual digits using an interactive GAN system based on the MNIST dataset.

            ### Context for you:
            This system shows images of handwritten digits (0–9) and can transform one digit into another using a Conditional Counterfactual GAN.  
            Users can optionally control *which image regions (patches)* are allowed to change. A reference grid image, "patch_grid.png", shows how the patches are numbered.

            ### Your behavior:
            1. **Welcome the user warmly** and explain briefly what the system can do:
            - That they can explore how digits can be transformed into others.
            - That they can control which regions (patches) are allowed to change.
            - Mention that a patch reference image ("patch_grid.png") is shown for guidance.

            2. **Guide them step by step**:
            - Ask: “Which digit would you like to inspect first?”  
            - Wait for the user's response (`base digit`).
            - Then confirm: “Okay, this is a digit X. What digit do you want it to transform into?”  
            - Wait for the user's response (`target digit`).
            - Then ask: “Which patches should be allowed to change? (If unsure, I’ll use all patches by default.)”

            3. **Once the user answers all questions**, summarize their intent in **one JSON object only**, with the following format:

            ```json
            {{
                "base": <integer between 0–9 or null>,
                "target": <integer between 0–9 or null>,
                "patches": [list of integers between 0 and {total_patches-1}]
            }}
            ```

            Current conversation:
                        {history}
                        Human: {input}
                        AI Assistant:
            """
            
            prompt = PromptTemplate(
                input_variables=["history", "input"], 
                template=template
            )
            
            # Set up memory for conversation history
            memory = ConversationBufferMemory(return_messages=True)
            
            # Initialize the Gemini model
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                top_p=0.95,
                google_api_key=GENAI_API_KEY,
                convert_system_message_to_human=True
            )
            
            # Create conversation chain
            self.conversation = ConversationChain(
                llm=llm,
                memory=memory,
                prompt=prompt,
                verbose=False
            )
            
            logger.info("Gemini model successfully initialized.")
            
        except Exception as e:
            logger.error(f"Model Setup Error: {e}")
            raise
    
    def get_response(self, user_message, history):
        try:
            if not user_message.strip():
                return "Please enter a message to continue the conversation."
            
            # Get response from the model
            response = self.conversation.predict(input=user_message)
            
            # Update history
            history.append((user_message, response))
            
            return response
        except Exception as e:
            logger.error(f"Response Generation Error: {e}")
            return f"I'm having trouble processing your request. Please try again later. (Error: {type(e).__name__})"

# Initialize the chatbot
chatbot = GeminiChatbot()

# Create Gradio interface
def launch_interface():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("""
        # 🤖 Gemini AI Chatbot
        Have a conversation with Google's Gemini model powered by LangChain! Ask questions, seek advice, or just chat.
        """)
        
        chatbot_ui = gr.Chatbot(
            label="Conversation",
            height=600,
            bubble_full_width=False,
            avatar_images=(
                "https://api.dicebear.com/7.x/thumbs/svg?seed=user",
                "https://api.dicebear.com/7.x/thumbs/svg?seed=assistant"
            )
        )
        
        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                placeholder="Type your message here...",
                container=False,
                scale=9
            )
            submit = gr.Button("Send", variant="primary", scale=1)
        
        clear = gr.Button("Clear Conversation")
        
        # Event handlers
        history = []
        
        def respond(message, chat_history):
            bot_response = chatbot.get_response(message, history)
            chat_history.append((message, bot_response))
            return "", chat_history
        
        def clear_conversation():
            chatbot.conversation_history = []
            return [], []
        
        # Set up event listeners
        submit.click(respond, [msg, chatbot_ui], [msg, chatbot_ui])
        msg.submit(respond, [msg, chatbot_ui], [msg, chatbot_ui])
        clear.click(clear_conversation, None, [chatbot_ui, msg])
        
        gr.Markdown("""
        ### 💡 Tips
        - Be specific in your questions for better answers
        - You can ask follow-up questions to dive deeper into a topic
        - Type 'help' if you need assistance with using this chatbot
        """)
        
    return demo

if __name__ == "__main__":
    try:
        demo = launch_interface()
        demo.launch(share=True)
    except Exception as e:
        logger.critical(f"Application Error: {e}")

# # --- Helper functions ---
# def pick_sample_by_digit(dataset, digit: int) -> torch.Tensor:
#     indices = [i for i, t in enumerate(dataset.targets) if int(t) == int(digit)]
#     if not indices:
#         raise ValueError(f"No sample of digit {digit}")
#     idx = random.choice(indices)
#     img, label = dataset[idx]
#     return img.unsqueeze(0), int(label)

def make_mask_from_patch_list(x_batch, patch_size, allowed_patches):
    _, single_mask = eval.build_patch_mask_for_batch(
        x_batch, patch_size=patch_size, device=DEVICE,
        shared_per_batch=True, modifiable_patches=allowed_patches,
        return_single_mask=True, randomize_per_sample=False
    )
    return single_mask

def run_transformation(x, label, target_digit:int, patches:List[int]):
    if GLOB_GENERATOR is None or GLOB_CLASSIFIER is None:
        return None, None, "Generator or classifier not loaded."
    single_mask = make_mask_from_patch_list(x, PATCH_SIZE, patches)
    x_vis = ((x + 1.0)/2.0).detach().cpu()
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

# # --- Deterministic parser ---
# def deterministic_parse(user_text:str, total_patches:int):
#     out = {"base": None, "target": None, "patches": []}
#     digits = re.findall(r"\b([0-9])\b", user_text)
#     if len(digits) >= 1:
#         out["base"] = int(digits[0])
#     if len(digits) >= 2:
#         out["target"] = int(digits[1])
#     # parse patches
#     m = re.search(r"patch(?:es)?\s*[:=]?\s*([0-9,\-\s]+)", user_text, flags=re.IGNORECASE)
#     if m:
#         chunk = m.group(1)
#         parts = [p.strip() for p in re.split(r"[,\s]+", chunk) if p.strip()]
#         patches = []
#         for p in parts:
#             if re.match(r"^\d+-\d+$", p):
#                 a,b = [int(x) for x in p.split("-")]
#                 patches.extend(list(range(a, b+1)))
#             elif p.isdigit():
#                 patches.append(int(p))
#         out["patches"] = sorted(set([p for p in patches if 0 <= p < total_patches]))
#     return out

# # --- LLM parser fallback ---
# def llm_parse_intent(user_text:str, total_patches:int):
#     prompt = f"""
# You are a friendly assistant helping a user transform MNIST digits using CounterGAN.
# The user said: "{user_text}"
# Infer:
# - base digit (0-9 or random)
# - target digit (0-9)
# - allowed patches (list, default all if missing)
# Return exactly one JSON object:
# {{"base": <0-9>, "target": <0-9>, "patches": [0,1,...]}}
# """
#     if GEMINI_MODEL is None:
#         raise RuntimeError("Gemini not configured for LLM parsing.")
#     raw = GEMINI_MODEL.generate(prompt=prompt, temperature=0.0, max_output_tokens=256)
#     # extract JSON
#     import re, json
#     txt = str(raw)
#     m = re.search(r"(\{.*\})", txt, flags=re.DOTALL)
#     if not m:
#         raise ValueError(f"LLM output invalid: {txt}")
#     parsed = json.loads(m.group(1))
#     if "patches" not in parsed or not parsed["patches"]:
#         parsed["patches"] = list(range(total_patches))
#     return parsed

# def assistant(user_text: str, state: dict):
#     total_patches = (28 // PATCH_SIZE) ** 2

#     # Step 0: welcome
#     if state.get("step", 0) == 0:
#         state.update({"step": 1})
#         return (
#             "Welcome! You can explore MNIST digits and transform them into other digits. "
#             "You can also control which regions (patches) are allowed to change. "
#             "A patch reference image ('patch_grid.png') will be shown for guidance.\n\n"
#             "Which digit would you like to inspect first?",
#             state, None, None, None, None
#         )

#     # Step 1: get base digit
#     if state.get("step") == 1:
#         m = re.search(r"\b([0-9])\b", user_text)
#         state["base"] = int(m.group(1)) if m else None
#         state["step"] = 2
#         return (
#             f"Okay, this is digit {state['base']}. What digit do you want it to transform into?",
#             state, None, None, None, None
#         )

#     # Step 2: get target digit
#     if state.get("step") == 2:
#         m = re.search(r"\b([0-9])\b", user_text)
#         state["target"] = int(m.group(1)) if m else None
#         state["step"] = 3
#         return (
#             f"Which patches should be allowed to change? (If unsure, I will use all patches by default. Patches range 0–{total_patches-1})",
#             state, None, None, None, None
#         )

#     # Step 3: get patches + run transformation
#     if state.get("step") == 3:
#         # parse patches
#         patches = []
#         m = re.findall(r"\d+", user_text)
#         if m:
#             patches = [int(p) for p in m if 0 <= int(p) < total_patches]
#         state["patches"] = patches if patches else list(range(total_patches))

#         # Pick MNIST sample
#         base = state["base"] if state["base"] is not None else random.choice(range(10))
#         target = state["target"] if state["target"] is not None else base
#         x, label = pick_sample_by_digit(mnist_test, base)
#         x_vis = ((x + 1.0) / 2.0).detach().cpu()

#         # Save original + patch grid
#         orig_path = os.path.join(SAVE_DIR, "original_sample.png")
#         patch_grid_path = os.path.join(SAVE_DIR, "patch_grid.png")
#         torchvision.utils.save_image(x_vis, orig_path)
#         eval.visualize_patch_grid(x_vis[0], PATCH_SIZE, patch_grid_path)

#         # Run transformation
#         out_img, msg = run_transformation(x, label, target, state["patches"])

#         # Prepare reply
#         reply = f"Transformed digit {base} → {target}. See images and heatmap above."

#         # Reset state for next session
#         state.update({"step": 0, "base": None, "target": None, "patches": []})

#         return reply, state, orig_path, patch_grid_path, out_img




# # --- Gradio UI ---
# with gr.Blocks(title="MNIST CC-GAN + Gemini chatbot assistant") as demo:
#     gr.Markdown("## MNIST CC-GAN demo with a friendly assistant")
#     with gr.Row():
#         user_input = gr.Textbox(lines=2, placeholder="Type something like 'transform 3 to 7, patches 0-3'", label="Your request")
#         assistant_reply_box = gr.Textbox(label="Chatbot assistant", interactive=False)
#         state = gr.State(value={"step": 0, "base": None, "target": None, "patches": []})
#     with gr.Row():
#         send_btn = gr.Button("Send")
#         orig_img = gr.Image(label="Original digit", type="filepath", height=280, width=280)
#         grid_img = gr.Image(label="Patch grid", type="filepath", height=280, width=280)
#     with gr.Row():
#         result_img = gr.Image(label="Transformed digit", type="filepath")
#     send_btn.click(
#         assistant,
#         inputs=[user_input, state],
#         outputs=[assistant_reply_box, state, orig_img, grid_img, result_img]
#     )

# # TODO: smooth gradio interface from github


# if __name__ == "__main__":
#     try:
#         load_models(generator_path=cfg.generator_path, classifier_path=cfg.classifier_path)
#     except Exception as e:
#         print(f"Failed to load models at startup: {e}")
#     demo.launch(server_name="0.0.0.0")
