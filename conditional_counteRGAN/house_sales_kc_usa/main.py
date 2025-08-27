#!/usr/bin/env python3
import torch
from config import config
from data_utils import load_and_preprocess
from models.nn_classifier import NNClassifier
from trainer import train_countergan
from eval_utils import evaluate_pipeline

# Load and preprocess
X_train, X_test, y_train, y_test = load_and_preprocess(config['data_path'], config['seed'])

device = config['cuda']

# Load classifier
checkpoint = torch.load(config['clf_model_path'], map_location=device)
clf = NNClassifier(config['input_dim']).to(device)
clf.load_state_dict(checkpoint['model_state_dict'])
clf.eval()

# Train CounterGAN
G = train_countergan(config, X_train, y_train, clf)

# evaluate & save everything
metrics_df = evaluate_pipeline(G, clf, X_test, y_test, config)
print(metrics_df)