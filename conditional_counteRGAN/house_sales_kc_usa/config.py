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
    'bins': None,  # class bins, to be set after loading data
    'gumbel_tau': 0.5,  # Gumbel-Softmax temperature
    'scaler': None,  # to be set after loading data
    'openai_api_key': None,

    # Classifier-specific outputs
    'clf_dir': clf_dir,
    'clf_model_path': 'clf_model.pt',
    'generator_path': 'generator_model.pt',
    'clf_loss_plot': os.path.join(clf_dir, 'loss_plot.png'),

    'immutable_features': ["lat", "long", "yr_built", "yr_renovated"], # "yr_renovated"
    'feature_names': [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "floors", "waterfront", "view", "condition", "grade",
        "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
        "lat", "long", "sqft_living15", "sqft_lot15"
    ]

    
}

config['immutable_idx'] = [config['feature_names'].index(f) for f in config['immutable_features']]
config['categorical_features'] = ["bedrooms", "bathrooms", "floors", "waterfront", "view", "condition", "grade"]
config['categorical_idx'] = [config['feature_names'].index(f) for f in config['categorical_features']]
config['categorical_info'] = {
    # bedrooms (0..8+)
    config['feature_names'].index("bedrooms"): {
        "n": 9,
        "raw_values": [0, 1, 2, 3, 4, 5, 6, 7, 8]
    },

    # bathrooms (0.00–8.00)
    config['feature_names'].index("bathrooms"): {
        "n": 30,
        "raw_values": sorted([
            0.00, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50,
            2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00,
            5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.50, 7.75, 8.00
        ])
    },

    config['feature_names'].index("floors"):     {"n": 6,  "raw_values": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]},
    config['feature_names'].index("waterfront"): {"n": 2,  "raw_values": [0, 1]},
    config['feature_names'].index("view"):       {"n": 5,  "raw_values": [0, 1, 2, 3, 4]},
    config['feature_names'].index("condition"):  {"n": 5,  "raw_values": [1, 2, 3, 4, 5]},
    config['feature_names'].index("grade"):      {"n": 13, "raw_values": list(range(1, 14))}
}

# Continuous features are the rest
config['continuous_idx'] = [i for i in range(config['input_dim']) if i not in config['categorical_info']]
