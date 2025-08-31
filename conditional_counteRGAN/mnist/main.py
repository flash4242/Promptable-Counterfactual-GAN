#!/usr/bin/env python
import torch
import os
from config import Config
from data_utils import get_dataloaders
from models.classifier import CNNClassifier
from models.generator import ResidualGenerator
from models.discriminator import Discriminator
from trainer import train_classifier, train_countergan
from eval_utils import evaluate_classifier, visualize_counterfactual_grid, evaluate_counterfactuals

config = Config()
device = config.device
os.makedirs(config.save_dir, exist_ok=True)

train_loader, valid_loader, test_loader, full_dataset = get_dataloaders(config.batch_size, config.num_workers, config.data_dir)

classifier = CNNClassifier(num_classes=config.num_classes).to(device)
generator = ResidualGenerator(img_shape=config.img_shape, num_classes=config.num_classes).to(device)
discriminator = Discriminator(img_shape=config.img_shape, num_classes=config.num_classes).to(device)

classifier_path = config.classifier_path

if os.path.exists(classifier_path):
    print(f"Loading pretrained classifier from {classifier_path}...")
else:
    print("Pretraining classifier...")
    train_classifier(classifier, train_loader, valid_loader, config, device)
classifier.load_state_dict(torch.load(classifier_path, map_location=device))

evaluate_classifier(classifier, test_loader, device, save_dir=config.save_dir)

print("Training CounterGAN...")
train_countergan(generator, discriminator, classifier, train_loader, config, device)

visualize_counterfactual_grid(generator, classifier, full_dataset, device,
                                save_path=os.path.join(config.save_dir, "cf_grid.png"))

print("Evaluating counterfactuals...")
for x, y in test_loader:
    # pick a random *different* target for each y
    y_target = torch.randint(0, 10, y.shape)
    mask = (y_target == y)
    y_target[mask] = (y_target[mask] + 1) % 10   # ensure different target

    cf_metrics, x_cf = evaluate_counterfactuals(generator, classifier, x, y, y_target, device)
    print(cf_metrics)
    break   # only run once for inspection