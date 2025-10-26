#!/usr/bin/env python3
import os
import re
import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd

from config import config
from data_utils import load_and_preprocess
from eval_utils import build_counterfactuals

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

device = config.get("cuda", "cpu")
X_train, X_test, y_train, y_test = load_and_preprocess(config["data_path"], config)
scaler = config['scaler']
num_classes = config["num_classes"]
feature_names = config.get("feature_names", [f"feat_{i}" for i in range(X_train.shape[1])])

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


def describe_classes(y):
    bins = config['bins']
    summary = {f"Class {i}": f"${bins[i]:,.0f} - ${bins[i+1]:,.0f}" for i in range(len(bins) - 1)}
    return summary


def get_random_sample_from_class(source_class):
    idxs = np.where(y_train == int(source_class))[0]
    if len(idxs) == 0:
        raise ValueError(f"No samples found for class {source_class}")
    choice = np.random.choice(idxs)
    x_sample = X_train[choice].astype(np.float32)
    return x_sample, int(choice)


def parse_natural_instruction(instruction, feature_names, immutable_idx):
    mask = np.zeros(len(feature_names), dtype=np.float32)

    if client:
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
        instruction = (instruction or "").lower()
        allowed = []
        for feat in feature_names:
            if re.search(rf"\b{re.escape(feat.lower())}\b", instruction):
                allowed.append(feat)
        synonyms = {
            "bed": "bedrooms", "bath": "bathrooms", "living": "sqft_living",
            "lot": "sqft_lot", "renovation": "yr_renovated", "grade": "grade",
            "view": "view", "condition": "condition", "floor": "floors"
        }
        for word, feat in synonyms.items():
            if re.search(rf"\b{re.escape(word)}\b", instruction):
                allowed.append(feat)

    for i, feat in enumerate(feature_names):
        if i in immutable_idx:
            mask[i] = 0
        elif feat in allowed:
            mask[i] = 1
        else:
            mask[i] = 0
    return mask, sorted(list(set(allowed)))


def show_class_summary():
    return describe_classes(y_train)


def show_sample(source_class):
    try:
        x_sample, idx = get_random_sample_from_class(source_class)
        x_tensor = torch.tensor(x_sample, dtype=torch.float32, device=device).unsqueeze(0)
        preds = clf(x_tensor).softmax(1).detach().cpu().numpy().squeeze()

        try:
            denorm = scaler.inverse_transform(x_sample.reshape(1, -1)).squeeze()
            df_table = pd.DataFrame({"feature": feature_names, "value": denorm})
        except Exception:
            df_table = pd.DataFrame({"feature": feature_names, "value (norm)": x_sample})

        table_html = df_table.to_html(index=False)

        pred_str = "\n".join([f"Class {i}: {p:.3f}" for i, p in enumerate(preds)])
        state_payload = {"x_sample": x_sample.tolist(), "idx": int(idx), "source_class": int(source_class)}
        return state_payload, table_html, pred_str

    except Exception as e:
        import traceback
        print("Error in show_sample():", traceback.format_exc())
        return None, None, f"Error: {e}"


def generate_counterfactual(x_state, source_class, target_class, user_instruction):
    try:
        if not x_state:
            return None, "No sample selected. Please get a random sample first.", None

        x_sample = np.array(x_state["x_sample"], dtype=np.float32)
        mask, allowed = parse_natural_instruction(user_instruction, feature_names, config.get("immutable_idx", []))

        allowed = set(allowed)
        per_request_immutable = [i for i, feat in enumerate(feature_names) if feat not in allowed]
        cfg = dict(config)
        cfg['immutable_idx'] = per_request_immutable
        cfg['scaler'] = scaler

        x_t = torch.tensor(x_sample, dtype=torch.float32, device=device).unsqueeze(0)
        target_vec = torch.tensor([int(target_class)], dtype=torch.long, device=device)
        target_onehot = F.one_hot(target_vec, num_classes).float()

        with torch.no_grad():
            masked_residual, x_cf = build_counterfactuals(G, x_t, target_onehot, cfg)

        x_cf_np = x_cf.cpu().numpy().squeeze()

        with torch.no_grad():
            orig_probs = F.softmax(clf(torch.tensor(x_sample, dtype=torch.float32, device=device).unsqueeze(0)), dim=1).cpu().numpy().squeeze()
            cf_probs = F.softmax(clf(torch.tensor(x_cf_np, dtype=torch.float32, device=device).unsqueeze(0)), dim=1).cpu().numpy().squeeze()

        try:
            orig_denorm = scaler.inverse_transform(x_sample.reshape(1, -1)).squeeze()
            cf_denorm = scaler.inverse_transform(x_cf_np.reshape(1, -1)).squeeze()
            abs_delta_denorm = np.abs(cf_denorm - orig_denorm)
            if hasattr(scaler, 'data_max_') and hasattr(scaler, 'data_min_'):
                feature_range = (scaler.data_max_ - scaler.data_min_).astype(float)
            else:
                feature_range = np.ones_like(abs_delta_denorm)
            pct_of_range = (abs_delta_denorm / (feature_range + 1e-12)) * 100.0
        except Exception:
            orig_denorm = x_sample
            cf_denorm = x_cf_np
            abs_delta_denorm = np.abs(cf_denorm - orig_denorm)
            pct_of_range = abs_delta_denorm * 100.0

        df = pd.DataFrame({
            "feature": feature_names,
            "original": orig_denorm,
            "counterfactual": cf_denorm,
            "change": abs_delta_denorm,
            "percentage of change": pct_of_range
        })

        eps = 1e-3
        df['changed'] = (df['abs_delta'] > eps)

        def row_to_html(row):
            style = 'background-color: #2a9d8f; color: white;' if row['changed'] else ''
            return f"<tr style=\"{style}\"><td>{row['feature']}</td><td>{row['orig']:.4f}</td><td>{row['cf']:.4f}</td><td>{row['abs_delta']:.4f}</td><td>{row['pct_of_range']:.2f}%</td></tr>"

        header = "<table border=1 cellpadding=5><tr><th>feature</th><th>orig</th><th>cf</th><th>abs_delta</th><th>pct_of_range</th></tr>"
        rows_html = "".join([row_to_html(r) for _, r in df.iterrows()])
        table_html = header + rows_html + "</table>"

        fig, ax = plt.subplots(figsize=(6, max(3, len(feature_names)*0.2)))
        diffs = x_cf_np - x_sample
        ax.barh(feature_names, diffs)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Counterfactual Δ (Class {int(source_class)} → {int(target_class)})")
        plt.tight_layout()

        summary_lines = [f"Modified features (allowed by instruction): {', '.join(sorted(list(allowed))) if allowed else 'None'}",
                         f"Classifier orig probs: {np.round(orig_probs, 3).tolist()}",
                         f"Classifier cf probs:   {np.round(cf_probs, 3).tolist()}"]
        summary_text = "\n".join(summary_lines)

        return fig, summary_text, table_html

    except Exception as e:
        import traceback
        print("Error in generate_counterfactual():", traceback.format_exc())
        return None, f"Error: {e}", None


with gr.Blocks() as demo:
    gr.Markdown("## 🏠 CounteRGAN Demo — King County Housing Data")

    gr.Markdown("### 1️⃣ Dataset Overview")
    class_summary = gr.JSON(label="Class Counts")
    show_btn = gr.Button("Show Class Overview")
    show_btn.click(fn=show_class_summary, outputs=[class_summary])

    gr.Markdown("### 2️⃣ Choose a source class to sample from")
    src_class = gr.Number(label="Source class (0–3)", value=0)
    x_state = gr.State()
    src_table = gr.HTML(label="Selected House (denormalized features)")
    src_pred_txt = gr.Textbox(label="Source sample classifier probs", interactive=False)

    get_sample_btn = gr.Button("Get Random Sample")
    get_sample_btn.click(fn=show_sample, inputs=src_class, outputs=[x_state, src_table, src_pred_txt])

    gr.Markdown("### 3️⃣ Generate Counterfactual via Natural Instruction")
    target_class = gr.Number(label="Target class (0–3)", value=1)
    instruction = gr.Textbox(
        label="Instruction (e.g., 'increase grade and bathrooms, change view, keep location fixed')",
        placeholder="Describe which features can change...",
    )
    cf_plot = gr.Plot(label="Counterfactual Changes (Δ per feature)")
    cf_summary = gr.Textbox(label="Summary", interactive=False)
    cf_table = gr.HTML(label="Counterfactual Table (denormalized)")
    gen_btn = gr.Button("Generate Counterfactual")

    gen_btn.click(
        fn=generate_counterfactual,
        inputs=[x_state, src_class, target_class, instruction],
        outputs=[cf_plot, cf_summary, cf_table],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)