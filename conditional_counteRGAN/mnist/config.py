import torch

class Config:
    batch_size = 128
    num_workers = 4

    num_epochs_gan = 20
    num_epochs_clf = 10
    cls_lr = 1e-3
    d_lr = 1e-5
    g_lr = 5e-5
    lambda_adv = 1.0
    lambda_cls = 1.0
    lambda_reg = 2.5
    lambda_mask = 2.0  # weight for mask penalty (to encourage sparse changes)
    patch_size = 7  # size of square patch mask to modify (5x5)
    num_modifiable_patches = 10  # number of patches that can be modified
    min_modifiable_patches = 6     # prevent degenerate cases
    max_modifiable_patches = 15    # optional cap
    user_input_patches = [1, 5, 10, 12, 13, 14]      # e.g., [3,4,8,12] when in interactive mode
    latent_dim = 100
    img_shape = (1, 28, 28)
    num_classes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./results"
    classifier_path = f"{save_dir}/best_classifier.pt"
    generator_path = f"{save_dir}/generator.pt"
    data_dir = "/mnt/data"
