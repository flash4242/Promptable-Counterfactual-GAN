import os
import torch

results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(results_dir, exist_ok=True)
mask_dir = os.path.join(results_dir, "mask_analysis")
os.makedirs(mask_dir, exist_ok=True)

config = {
    'cuda': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_path': 'kc_house_data.csv',
    'batch_size': 128,
    'input_dim': 17,
    'lr_G': 1e-3,
    'lr_D': 1e-3,
    'lambda_cls': 2.0,
    'lambda_reg': 1.0,
    'lambda_mask': 1.0,
    'hidden_dim': 32,
    'output_dim': 1,
    'epochs': 40,
    'lr': 5e-4,
    'eps': 1e-8,  # small value to avoid division by zero
    'seed': 42,
    'out_dir': results_dir,
    'out_dir_mask': mask_dir,
    'clf_model_path': 'clf_model.pth'
}