#!/usr/bin/env python3
import os
import re
import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from tabulate import tabulate

from config import config
from data_utils import load_and_preprocess

# ===== Optional OpenAI API =====
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", config.get("openai_api_key", None))
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        client = None
except Exception:
    client = None


# ===== Load Data & Models =====
from models.nn_classifier import NNClassifier
from models.generator import ResidualGenerator

device = config["cuda"]
X_train, X_test, y_train, y_test, scaler = load_and_preprocess(config["data_path"])
config["scaler"] = scaler
num_classes = config["num_classes"]

clf = NNClassifier(config['input_dim'], output_dim=config['num_classes']).to(device)
clf.load_state_dict(torch.load(config["clf_model_path"], map_location=device))
clf.eval()

G = ResidualGenerator(
    config['input_dim'], config['hidden_dim'], num_classes,
    continuous_idx=config['continuous_idx'],
    categorical_info={k: {"n": v["n"], "raw_values": v["raw_values"]} for k, v in config["categorical_info"].items()},
    tau=config['gumbel_tau']
).to(device)
G.load_state_dict(torch.load(config["generator_path"], map_location=device))
G.eval()


# ===== Utilities =====
def describe_classes(y):
    """Summarize quartile ranges."""
    unique, counts = np.unique(y, return_counts=True)
    summary = {f"Class {int(u)}": int(c) for u, c in zip(unique, counts)}
    return summary


def plot_class_ranges(y):
    """Show class distribution."""
    plt.figure(figsize=(6, 4))
    plt.hist(y, bins=np.arange(-0.5, max(y)+1.5, 1), rwidth=0.8)
    plt.xticks(range(int(max(y)) + 1))
    plt.xlabel("Class (Quartile)")
    plt.ylabel("Count")
    plt.title("House Price Quartile Classes")
    plt.tight_layout()
    return plt.gcf()


def parse_natural_instruction(instruction, feature_names, immutable_idx):
    """
    Parse user instruction into a binary mask.
    Fallback: regex-based if LLM is unavailable.
    """
    mask = np.zeros(len(feature_names), dtype=np.float32)

    if client:
        # Use GPT parsing
        prompt = f"""
        You are a system that selects which housing features can be modified
        based on a user's natural-language instruction. The available features are:
        {feature_names}.
        Immutable features: {', '.join([feature_names[i] for i in immutable_idx])}.
        Return a Python list of feature names that the user allows to modify.
        Instruction: "{instruction}"
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "Return valid JSON only."},
                          {"role": "user", "content": prompt}],
                temperature=0,
            )
            text = response.choices[0].message.content
            allowed = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
            allowed = [a[0] or a[1] for a in allowed]
        except Exception as e:
            print("LLM parsing failed, fallback to regex:", e)
            allowed = []
    else:
        # Regex fallback
        instruction = instruction.lower()
        allowed = []
        for feat in feature_names:
            if re.search(rf"\b{feat.lower()}\b", instruction):
                allowed.append(feat)
        # simple synonyms
        synonyms = {
            "bed": "bedrooms", "bath": "bathrooms", "living": "sqft_living",
            "lot": "sqft_lot", "renovation": "yr_renovated", "grade": "grade",
            "view": "view", "condition": "condition", "floor": "floors"
        }
        for word, feat in synonyms.items():
            if re.search(rf"\b{word}\b", instruction):
                allowed.append(feat)

    # Convert to mask
    for i, feat in enumerate(feature_names):
        if i in immutable_idx:
            mask[i] = 0
        elif feat in allowed:
            mask[i] = 1
    return mask, allowed


# ===== Core Logic =====
def show_class_summary():
    return plot_class_ranges(y_train), describe_classes(y_train)

def show_sample(source_class):
    try:
        x_sample, y_sample = get_random_sample_from_class(source_class)
        x_tensor = torch.tensor(x_sample, dtype=torch.float32, device=device).unsqueeze(0)
        preds = clf(x_tensor).softmax(1).detach().cpu().numpy().squeeze()

        # --- make a pretty table of feature:value pairs ---
        df = pd.DataFrame([x_sample], columns=feature_names)
        table_html = df.to_html(index=False)

        pred_str = "\n".join([f"Class {i}: {p:.3f}" for i, p in enumerate(preds)])
        return table_html, pred_str

    except Exception as e:
        import traceback
        print("Error in show_sample():", traceback.format_exc())
        return None, f"Error: {e}"


def generate_counterfactual(x_sample, source_class, target_class, user_instruction):
    x_sample = np.array(x_sample, dtype=np.float32)
    mask, allowed = parse_natural_instruction(user_instruction, config["feature_names"], config["immutable_idx"])
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

    x_t = torch.tensor(x_sample, dtype=torch.float32, device=device).unsqueeze(0)
    target_vec = torch.tensor([target_class], dtype=torch.long, device=device)
    target_onehot = F.one_hot(target_vec, num_classes).float()

    with torch.no_grad():
        _, masked_residual = G(x_t, target_onehot, mask=mask_t)
        x_cf = x_t + masked_residual

    x_cf_np = x_cf.cpu().numpy().squeeze()

    # visualize
    fig, ax = plt.subplots(figsize=(6, 4))
    diffs = x_cf_np - x_sample
    ax.barh(config["feature_names"], diffs)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Counterfactual Δ (Class {source_class} → {target_class})")
    plt.tight_layout()

    return fig, f"Modified features: {', '.join(allowed) if allowed else 'None'}"


# ===== Gradio App =====
with gr.Blocks() as demo:
    gr.Markdown("## 🏠 CounterGAN Demo — King County Housing Data")

    # STEP 1
    gr.Markdown("### 1️⃣ Dataset Overview")
    plot_out = gr.Plot(label="Class Distribution")
    class_summary = gr.JSON(label="Class Counts")
    show_btn = gr.Button("Show Class Overview")
    show_btn.click(fn=show_class_summary, outputs=[plot_out, class_summary])

    # STEP 2
    gr.Markdown("### 2️⃣ Choose a source class to sample from")
    src_class = gr.Number(label="Source class (0–3)", value=0)
    x_state = gr.State()
    src_pred_txt = gr.HTML(label="Selected House (features)")

    get_sample_btn = gr.Button("Get Random Sample")
    get_sample_btn.click(fn=show_sample, inputs=src_class, outputs=[x_state, src_pred_txt])

    # STEP 3
    gr.Markdown("### 3️⃣ Generate Counterfactual via Natural Instruction")
    target_class = gr.Number(label="Target class (0–3)", value=1)
    instruction = gr.Textbox(
        label="Instruction (e.g., 'increase grade and bathrooms, change view, keep location fixed')",
        placeholder="Describe which features can change...",
    )
    cf_plot = gr.Plot(label="Counterfactual Changes (Δ per feature)")
    cf_summary = gr.Textbox(label="Summary", interactive=False)
    gen_btn = gr.Button("Generate Counterfactual")

    gen_btn.click(
        fn=generate_counterfactual,
        inputs=[x_state, src_class, target_class, instruction],
        outputs=[cf_plot, cf_summary],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
