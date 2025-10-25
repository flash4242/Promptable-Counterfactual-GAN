#!/usr/bin/env python3
import os
import json
import re
import random
import torch
import torchvision
from torchvision import transforms, datasets
import gradio as gr
import numpy as np
from typing import Tuple, List, Optional
from config import Config as cfg

# optional: LLM client
try:
    import openai
except Exception:
    openai = None

# Import your eval utils (assumes eval_utils.py is in same folder / on PYTHONPATH)
import eval_utils as eval

# --- Configuration (adjust or read from config.py) ---
PATCH_SIZE = cfg.patch_size                 # your working patch size (7 -> 4x4 grid)
MIN_MODIFIABLE = cfg.min_modifiable_patches                # minimum patches if user doesn't choose enough
MAX_MODIFIABLE = cfg.max_modifiable_patches             # optional cap (None -> half)
DEVICE = cfg.device

# MNIST dataset (unpadded, same transforms your pipeline expects)
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1,1]
    ])

mnist_test = datasets.MNIST(cfg.data_dir, train=False, transform=transform, download=False)

# Utility to pick a sample for a given source digit
def pick_sample_by_digit(dataset, digit: int, seed: Optional[int]=None) -> Tuple[torch.Tensor, int]:
    if seed is not None:
        random.seed(seed)
    indices = [i for i, t in enumerate(dataset.targets) if int(t) == int(digit)]
    if not indices:
        raise ValueError("No sample of that digit found.")
    idx = random.choice(indices)
    img, label = dataset[idx]
    # img is normalized (-1,1) as your pipeline expects
    return img.unsqueeze(0), int(label)        # shape (1,1,28,28), label

# Simple fallback parser for user free-text: extracts first digit as target and list of ints as patches
def parse_user_input_fallback(text: str, total_patches: int) -> Tuple[int, List[int], str]:
    text_lower = text.lower()
    # Try explicit 'target' or 'to' phrase first
    target_match = re.search(r"(?:target|to)\s*([0-9])", text_lower)
    if target_match:
        target = int(target_match.group(1))
    else:
        # fallback: first digit that appears
        digits = re.findall(r"\b([0-9])\b", text)
        target = int(digits[0]) if digits else None

    # Extract patch numbers after keyword 'patch' (if present)
    patch_section = re.split(r"patch(?:es)?", text_lower)
    if len(patch_section) > 1:
        patch_text = patch_section[1]
    else:
        patch_text = text_lower
    patch_candidates = re.findall(r"\b([0-9]{1,2})\b", patch_text)
    patches_int = [int(p) for p in patch_candidates if 0 <= int(p) < total_patches]

    if target is None:
        return None, [], "Couldn't detect a target digit. Please include a digit 0–9 as the desired target."
    if len(patches_int) == 0:
        return target, [], "No patch indices detected — choosing random allowed patches."
    return target, sorted(list(set(patches_int))), ""


# LLM parsing helper (optional). Returns (target:int, patches:list, message:str)
def parse_user_input_llm(user_text: str, total_patches: int) -> Tuple[Optional[int], List[int], str]:
    # If openai not available, fall back
    if openai is None or os.environ.get("OPENAI_API_KEY", "") == "":
        return parse_user_input_fallback(user_text, total_patches)

    openai.api_key = os.environ["OPENAI_API_KEY"]
    # prompt instructing the model to return JSON only
    prompt = f"""
You are a strict parser. Input is a short user instruction answering:
  "Into which digit do you want to transform this?" and "Which patch numbers are allowed (list)?"

Return a single JSON object on the reply ONLY with the keys:
  "target": integer between 0 and 9,
  "patches": array of integers (each between 0 and {total_patches-1}).

If the user does not provide patches, return an empty list for "patches".
If the target isn't present, set "target": null.

Do not include any commentary. Example output:
  {{ "target": 3, "patches": [1,5,6] }}
Now parse this user message:
---
{user_text}
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",   # choose a small/reasonable model; change as needed
            messages=[{"role":"user","content":prompt}],
            max_tokens=150,
            temperature=0.0
        )
        text = resp["choices"][0]["message"]["content"].strip()
        # Attempt to load JSON
        obj = json.loads(text)
        target = obj.get("target", None)
        patches = obj.get("patches", [])
        # validate
        patches_valid = [int(p) for p in patches if isinstance(p, (int,float)) or (isinstance(p,str) and p.isdigit())]
        patches_valid = [p for p in patches_valid if 0 <= p < total_patches]
        return int(target) if target is not None else None, patches_valid, ""
    except Exception as e:
        return parse_user_input_fallback(user_text, total_patches)  # fallback with helpful message

# Helper to build mask from explicit list of patches (returns batch_mask, single_mask)
def make_mask_from_patch_list(x_batch: torch.Tensor, patch_size: int, allowed_patches: List[int], device) -> Tuple[torch.Tensor, torch.Tensor]:
    # use your eval_utils function create behavior
    batch_mask, single_mask = eval.build_patch_mask_for_batch(x_batch, patch_size=patch_size, device=device,
                                                          shared_per_batch=True, modifiable_patches=allowed_patches,
                                                          return_single_mask=True, randomize_per_sample=False)
    return batch_mask, single_mask

# Main Gradio function: shows MNIST sample and patch-grid first
def get_sample_and_grid(source_req_text: str):
    """
    Interprets user natural language to pick a source digit.
    Returns: original_image (PIL/np), patch_grid_path, source_digit
    """
    # parse source digit (first digit found) else pick random
    m = re.search(r"\b([0-9])\b", source_req_text)
    if m:
        src = int(m.group(1))
    else:
        src = int(random.choice(range(10)))

    x, label = pick_sample_by_digit(mnist_test, src)
    save_dir = "./gradio_tmp"
    os.makedirs(save_dir, exist_ok=True)

    # visualize normalized → [0,1]
    x_vis = ((x + 1.0) / 2.0).detach().cpu()
    img_np = x_vis[0].squeeze().numpy()

    # save both the original image and the patch grid overlay
    orig_path = os.path.join(save_dir, "original_sample.png")
    patch_grid_path = os.path.join(save_dir, "patch_grid.png")

    torchvision.utils.save_image(x_vis, orig_path)  # save denormalized image
    eval.visualize_patch_grid(x_vis[0], patch_size=PATCH_SIZE, save_path=patch_grid_path)

    return orig_path, patch_grid_path, src


# Callback to handle user's NL answer (target + patches). This uses LLM parse if key present else fallback
def handle_user_answer(user_text: str, sample_source_img, sample_label, batch_tensor_path=None):
    """
    user_text: NL answer indicating target and patches
    sample_source_img, sample_label: the original chosen sample returned from get_sample_and_grid
    """
    # sample_source_img expected as torch Tensor in previous pipeline? We'll instead re-pick sample here to get tensors.
    # For simplicity we will pick a sample for sample_label:
    x, label = pick_sample_by_digit(mnist_test, sample_label)
    bs = 1
    total_patches = (28 // PATCH_SIZE) * (28 // PATCH_SIZE)

    # parse user_text with LLM (if available) else fallback
    target, patches, msg = parse_user_input_llm(user_text, total_patches)
    if target is None:
        return None, f"Could not parse target: {msg}"

    # if user provided zero patches, choose random between MIN and MAX
    if len(patches) == 0:
        max_p = MAX_MODIFIABLE if MAX_MODIFIABLE is not None else total_patches // 2
        k = random.randint(MIN_MODIFIABLE, max(MIN_MODIFIABLE, max_p))
        patches = sorted(random.sample(range(total_patches), k))

    # Validate patches
    patches = sorted([p for p in patches if 0 <= p < total_patches])
    if len(patches) < MIN_MODIFIABLE:
        return None, f"Too few allowed patches ({len(patches)}). Minimum required is {MIN_MODIFIABLE}."

    # Build masks (shared per batch since we have 1 sample)
    batch_mask, single_mask = make_mask_from_patch_list(x, PATCH_SIZE, patches, DEVICE)

    # Call your generator/classifier via save_user_modification_example (which uses generator+classifier)
    save_dir = "./gradio_tmp"
    os.makedirs(save_dir, exist_ok=True)
    # we must pass generator and classifier objects; assume user has loaded them into global variables below
    global GLOB_GENERATOR, GLOB_CLASSIFIER
    if GLOB_GENERATOR is None or GLOB_CLASSIFIER is None:
        return None, "Generator / classifier not loaded on server. Start the app with models available."

    # call the helper in eval_utils (it will save the image to disk)
    eval.save_user_modification_example(
        x_vis=((x + 1.0) / 2.0).detach().cpu(),
        simulated_patches=patches,
        generator=GLOB_GENERATOR,
        classifier=GLOB_CLASSIFIER,
        y_true=torch.tensor([label], device=DEVICE),
        y_target=torch.tensor([target], device=DEVICE),
        device=DEVICE,
        save_dir=save_dir,
        patch_size=PATCH_SIZE
    )

    # also produce the sample heatmap (the more detailed one) for convenience
    # build a batch mask and generate CF to build the sample heatmap images using make_and_save_heatmaps helpers:
    with torch.no_grad():
        raw_res, masked_res = GLOB_GENERATOR(x.to(DEVICE), torch.tensor([target], device=DEVICE), batch_mask.to(DEVICE))
        x_cf = torch.clamp(x.to(DEVICE) + masked_res, -1.0, 1.0)
    x_vis = ((x + 1.0) / 2.0).detach().cpu()
    xcf_vis = ((x_cf + 1.0) / 2.0).detach().cpu()
    mask_vis = batch_mask.detach().cpu()
    metrics = eval.compute_masked_metrics(raw_res, masked_res, x.to(DEVICE), x_cf, batch_mask.to(DEVICE),
                                      GLOB_CLASSIFIER, torch.tensor([label], device=DEVICE), torch.tensor([target], device=DEVICE), DEVICE)
    # save sample heatmap
    eval.make_and_save_heatmaps(x_vis, xcf_vis, mask_vis, metrics, save_dir=save_dir,
                             y_true=torch.tensor([label]), y_target=torch.tensor([target]),
                             classifier=GLOB_CLASSIFIER, device=DEVICE, max_samples=1)

    # Return file path(s) for Gradio to show
    return {
        "simulated_example": os.path.join(save_dir, "simulated_user_modification.png"),
        "sample_heatmap": os.path.join(save_dir, "sample_0_src{}_tgt{}.png".format(label, target)),
        "patch_grid": os.path.join(save_dir, "patch_grid.png")
    }, f"Done. transformed {label} → {target} using patches {patches} (parsed)."

# --- Globals for models (load your trained models here) ---
GLOB_GENERATOR = None
GLOB_CLASSIFIER = None

def load_models(generator_path: str, classifier_path: str, device=DEVICE):
    global GLOB_GENERATOR, GLOB_CLASSIFIER
    # You must import your actual generator/classifier classes
    from models.generator import ResidualGenerator
    from models.classifier import CNNClassifier
    # instantiate with same shapes
    GLOB_GENERATOR = ResidualGenerator(img_shape=(1,28,28)).to(device)
    GLOB_CLASSIFIER = CNNClassifier().to(device)
    if os.path.exists(generator_path):
        GLOB_GENERATOR.load_state_dict(torch.load(generator_path, map_location=device))
    if os.path.exists(classifier_path):
        GLOB_CLASSIFIER.load_state_dict(torch.load(classifier_path, map_location=device))
    GLOB_GENERATOR.eval()
    GLOB_CLASSIFIER.eval()
    print("Loaded models.")

# --- Build minimal Gradio interface ---
def build_app(generator_path: str, classifier_path: str):
    load_models(generator_path, classifier_path, DEVICE)

    with gr.Blocks() as demo:
        gr.Markdown("## CounterGAN LLM-enabled demo (minimal)\n"
                    "Enter something like: 'Show me a 7' then press `Get sample`. "
                    "Then tell the system: 'Make it a 3 and allow patches 1,5,6' in the second box.")

        with gr.Row():
            with gr.Column():
                src_text = gr.Textbox(label="Pick source digit (natural language, e.g. 'show me a 7')",
                                      value="show me a 7")
                btn_sample = gr.Button("Get sample")
            with gr.Row():
                sample_img = gr.Image(label="Original MNIST digit", type="filepath", width=224, height=224)
                patch_grid_display = gr.Image(label="Patch grid (patch indices)", type="filepath", width=224, height=224)
                chosen_label_display = gr.Textbox(label="Selected source label", interactive=False)

            with gr.Column():
                user_answer = gr.Textbox(label="Your target & allowed patches (NL)", value="Target 3 patches 1,5,6")
                btn_apply = gr.Button("Apply transformation (LLM parse)")
                output_text = gr.Textbox(label="Status / messages", interactive=False)
                out_sim_example = gr.Image(label="Simulated modification (allowed patches)")
                out_sample_heatmap = gr.Image(label="Sample heatmap (Original/CF/Residual/Mask)")

        # Handlers
        def on_get_sample(txt):
            orig_path, patch_grid_path, src = get_sample_and_grid(txt)
            chosen_label = str(src)
            return orig_path, patch_grid_path, chosen_label


        btn_sample.click(fn=on_get_sample, inputs=[src_text], outputs=[sample_img, patch_grid_display, chosen_label_display])

        def on_apply(ans_text, sample_label_str):
            try:
                sample_label = int(sample_label_str)
            except:
                return None, None, "No source sample chosen."

            results, msg = handle_user_answer(ans_text, None, sample_label)
            if results is None:
                return None, None, msg
            # return images
            return results["simulated_example"], results["sample_heatmap"], msg

        btn_apply.click(fn=on_apply, inputs=[user_answer, chosen_label_display],
                        outputs=[out_sim_example, out_sample_heatmap, output_text])

    return demo

# === ENTRYPOINT ===
if __name__ == "__main__":
    # paths to your trained weights (adjust)
    gen_path = cfg.generator_path
    cls_path = cfg.classifier_path

    demo = build_app(gen_path, cls_path)
    demo.launch(server_name="0.0.0.0", share=False)
