import torch

class Config:
    batch_size = 128
    num_workers = 4

    num_epochs_gan = 25
    num_epochs_clf = 10
    cls_lr = 1e-3
    d_lr = 5e-5
    g_lr = 1e-4
    lambda_adv = 1.0
    lambda_cls = 1.0
    lambda_reg = 0.03

    latent_dim = 100
    img_shape = (1, 28, 28)
    num_classes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./results"
    classifier_path = f"{save_dir}/best_classifier.pt"
    data_dir = "/mnt/data"

# --- robust classifier training options ---
    adv_training = True
    adv_eps = 0.25   # perturbation radius in normalized space (~[-1,1])
    adv_alpha = 0.05
    adv_steps = 7

    # augmentation intensity for classifier training
    aug_rot_deg = 25
    aug_translate = 0.12
    aug_shear = 10
    aug_perspective = 0.5
