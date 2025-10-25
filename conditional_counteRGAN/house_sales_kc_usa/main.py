#!/usr/bin/env python3
import torch
from config import config
from data_utils import load_and_preprocess
from models.nn_classifier import NNClassifier
from models.generator import ResidualGenerator
from trainer import train_countergan, train_classifier
from eval_utils import evaluate_classifier
import os
import numpy
from eval_utils import evaluate_pipeline

# Load and preprocess
X_train, X_test, y_train, y_test, scaler = load_and_preprocess(config['data_path'])
config['scaler'] = scaler  # set the scaler in config for denormalization in evaluation

device = config['cuda']

# Load classifier
classifier = NNClassifier(config['input_dim'], output_dim=config['num_classes']).to(device)
classifier_path = config['clf_model_path']

if os.path.exists(classifier_path):
    print(f"Loading pretrained classifier from {classifier_path}...")
else:
    print("Pretraining classifier...")
    train_classifier(X_train, X_test, y_train, y_test, scaler, config)

classifier.load_state_dict(torch.load(classifier_path, map_location=device))
classifier.eval()
for p in classifier.parameters():
    p.requires_grad = False

# Evaluate classifier on test set (uses evaluate_classifier from eval_utils)
metrics = evaluate_classifier(
    classifier,
    X_test,
    y_test,
    out_dir=config['clf_dir'],
    device=device,
    class_names=[f"Class {i}" for i in range(config['num_classes'])]
)

num_classes = int(numpy.unique(y_train).size)
generator = ResidualGenerator(
    config['input_dim'], config['hidden_dim'], num_classes,
    continuous_idx=config['continuous_idx'],
    categorical_info={k: {"n": v["n"], "raw_values": v["raw_values"]} for k, v in config["categorical_info"].items()},
    tau=config['gumbel_tau']
).to(device)

generator_path = config['generator_path']
if os.path.exists(generator_path):
    print(f"Loading pretrained generator from {generator_path}...")
else:
    print("Training CounterGAN and saving generator...")
    train_countergan(generator, config, X_train, y_train, classifier)

generator.load_state_dict(torch.load(generator_path, map_location=config['cuda']))
generator.eval()
for p in generator.parameters():
    p.requires_grad = False

# evaluate & save everything
metrics_df = evaluate_pipeline(generator, classifier, X_test, y_test, config)
print(metrics_df)
