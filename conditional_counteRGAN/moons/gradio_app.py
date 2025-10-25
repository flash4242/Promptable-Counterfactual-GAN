#!/usr/bin/env python3
import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data_utils import load_and_preprocess
from config import config
from eval_utils import plot_decision_boundaries_and_cfs
import os

# Load trained models (assumed available)
from models.nn_classifier import NNClassifier
from models.generator import ResidualGenerator

device = config["cuda"]

# ===== Load data =====
X_train, X_test, y_train, y_test = load_and_preprocess(config["seed"])
num_classes = len(np.unique(y_train))

# ===== Load models =====
clf = NNClassifier(config["input_dim"], config["hidden_dim"], num_classes).to(device)
clf.load_state_dict(torch.load(config["clf_model_path"], map_location=device))
clf.eval()

G = ResidualGenerator(config["input_dim"], config["hidden_dim"], num_classes).to(device)
G.load_state_dict(torch.load(config["generator_path"], map_location=device))
G.eval()


# ===== Helper: visualize dataset =====
def plot_dataset_overview():
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolor="k", alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Available Classes in Extended Moons Dataset")
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.tight_layout()
    return plt.gcf()


# ===== Helper: show sample and decision boundary =====
def show_sample_and_boundary(source_class):
    if source_class not in [0, 1, 2]:
        raise gr.Error("Please choose a valid class (0, 1, or 2).")

    # select one random sample from the given class
    idxs = np.where(y_test == source_class)[0]
    if len(idxs) == 0:
        raise gr.Error(f"No samples found for class {source_class}")
    idx = np.random.choice(idxs)
    x_sample = X_test[idx]

    # plot decision boundary
    x_min, x_max = X_test[:, 0].min() - 0.05, X_test[:, 0].max() + 0.05
    y_min, y_max = X_test[:, 1].min() - 0.05, X_test[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        Z = clf(torch.tensor(grid, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", alpha=0.5)
    plt.scatter(x_sample[0], x_sample[1], c="black", edgecolor="yellow", s=100, label="Selected sample")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Selected sample from class {source_class}")
    plt.legend()
    plt.tight_layout()
    return plt.gcf(), x_sample.tolist(), int(source_class)


# ===== Helper: generate CF =====
def generate_counterfactual(x_sample, source_class, target_class, mask_choice):
    x_sample = np.array(x_sample, dtype=np.float32)
    x_t = torch.tensor(x_sample, dtype=torch.float32, device=device).unsqueeze(0)
    target_vec = torch.tensor([target_class], dtype=torch.long, device=device)
    target_onehot = F.one_hot(target_vec, num_classes).float()

    mask_dict = {
        "both": torch.tensor([[1.0, 1.0]], device=device),
        "x_only": torch.tensor([[1.0, 0.0]], device=device),
        "y_only": torch.tensor([[0.0, 1.0]], device=device),
        "none": torch.tensor([[0.0, 0.0]], device=device),
    }
    mask = mask_dict.get(mask_choice, mask_dict["both"])

    with torch.no_grad():
        _, masked_residual = G(x_t, target_onehot, mask=mask)
        x_cf = x_t + masked_residual

    x_cf_np = x_cf.squeeze().cpu().numpy()

    # plot
    x_min, x_max = X_test[:, 0].min() - 0.05, X_test[:, 0].max() + 0.05
    y_min, y_max = X_test[:, 1].min() - 0.05, X_test[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        Z = clf(torch.tensor(grid, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", alpha=0.3, label="Data")
    plt.scatter(x_sample[0], x_sample[1], c="black", edgecolor="yellow", s=100, label="Original")
    plt.scatter(x_cf_np[0], x_cf_np[1], c="lime", edgecolor="black", s=100, label="Counterfactual")
    plt.arrow(x_sample[0], x_sample[1],
              x_cf_np[0] - x_sample[0],
              x_cf_np[1] - x_sample[1],
              color="green", alpha=0.7, head_width=0.01, length_includes_head=True)
    plt.title(f"Transformation {source_class} → {target_class} ({mask_choice})")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


# ===== Gradio App =====
with gr.Blocks() as demo:
    gr.Markdown("## Counterfactual Explorer for Extended Moons Dataset")

    gr.Markdown("### Step 1: View dataset and available classes")
    initial_plot = plot_dataset_overview()
    dataset_plot = gr.Plot(value=initial_plot, label="Dataset overview")


    gr.Markdown("### Step 2: Choose a source class and view a random sample")
    src_class = gr.Number(label="Source class (0, 1, or 2)", value=0)
    sample_plot = gr.Plot(label="Decision boundary and selected sample")
    x_sample_state = gr.State()
    src_state = gr.State()
    get_sample_btn = gr.Button("Get sample from this class")

    get_sample_btn.click(
        fn=show_sample_and_boundary,
        inputs=src_class,
        outputs=[sample_plot, x_sample_state, src_state],
    )

    gr.Markdown("### Step 3: Choose a target class and modifiable dimensions")
    target_class = gr.Number(label="Target class (0, 1, or 2)", value=1)
    mask_choice = gr.Radio(
        choices=["both", "x_only", "y_only", "none"],
        label="Allowed modifiable dimensions",
        value="both",
    )
    cf_plot = gr.Plot(label="Counterfactual transformation")
    gen_btn = gr.Button("Generate counterfactual")

    gen_btn.click(
        fn=generate_counterfactual,
        inputs=[x_sample_state, src_state, target_class, mask_choice],
        outputs=cf_plot,
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
