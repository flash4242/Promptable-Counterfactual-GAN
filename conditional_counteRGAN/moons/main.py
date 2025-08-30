#!/usr/bin/env python3
from config import config
from data_utils import load_and_preprocess
from trainer import train_classifier, train_countergan
from eval_utils import evaluate_pipeline

#!/usr/bin/env python3
import os
import torch
from config import config
from data_utils import load_and_preprocess
from models.nn_classifier import NNClassifier
from trainer import train_countergan
from eval_utils import evaluate_pipeline
import torch.optim as optim
import torch.nn as nn

def get_classifier(X_train, y_train, config):
    device = config['cuda']
    clf = NNClassifier(config['input_dim']).to(device)
    clf_path = config['clf_model_path']

    if os.path.exists(clf_path):
        print(f"Loading existing classifier from {clf_path}")
        clf.load_state_dict(torch.load(clf_path, map_location=device))
        clf.eval()
        return clf

    print("Training new classifier...")
    opt = optim.Adam(clf.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)

    for _ in range(1000):
        preds = clf(X_t)
        loss = loss_fn(preds, y_t)
        opt.zero_grad(); loss.backward(); opt.step()

    os.makedirs(os.path.dirname(clf_path), exist_ok=True)
    torch.save(clf.state_dict(), clf_path)
    print(f"Saved classifier to {clf_path}")
    return clf

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess(config['seed'])
    clf = get_classifier(X_train, y_train, config)
    G = train_countergan(config, X_train, y_train, clf)
    metrics_df = evaluate_pipeline(G, clf, X_test, y_test, config)
    print(metrics_df)
