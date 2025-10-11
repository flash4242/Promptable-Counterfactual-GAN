import os
import torch

root_dir = os.getcwd()
results_dir = os.path.join(root_dir, "results")
os.makedirs(results_dir, exist_ok=True)

clf_dir = os.path.join(results_dir, "classifier_eval")
mask_dir = os.path.join(results_dir, "mask_analysis")
os.makedirs(clf_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

config = {
    'cuda': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_path': 'kc_house_data.csv',
    'batch_size': 128,
    'input_dim': 17,
    'num_classes': 4,
    'lr_G': 1e-3,
    'lr_D': 1e-3,
    'lambda_cls': 2.0,
    'lambda_reg': 1.0,
    'lambda_mask': 1.0,
    'hidden_dim': 32,
    'output_dim': 1,
    'epochs': 50,
    'lr': 5e-4,
    'eps': 1e-8,
    'seed': 42,
    'out_dir': results_dir,
    'out_dir_mask': mask_dir,

    # Classifier-specific outputs
    'clf_dir': clf_dir,
    'clf_model_path': 'clf_model.pt',
    'clf_loss_plot': os.path.join(clf_dir, 'loss_plot.png'),

    'immutable_features': ["lat", "long"], # , "yr_built", "yr_renovated"
    'feature_names': [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "floors", "waterfront", "view", "condition", "grade",
        "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
        "lat", "long", "sqft_living15", "sqft_lot15"
    ]
}

config['immutable_idx'] = [config['feature_names'].index(f) for f in config['immutable_features']]
