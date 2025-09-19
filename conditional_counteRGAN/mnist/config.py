import torch

class Config:
    batch_size = 128
    num_workers = 4

    num_epochs_gan = 10
    num_epochs_clf = 10
    cls_lr = 1e-3
    d_lr = 1e-4
    g_lr = 3e-4
    lambda_cls = 1.0
    lambda_reg = 5.0
    lambda_gp = 2.0

    latent_dim = 100
    img_shape = (1, 28, 28)
    num_classes = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./results"
    classifier_path = f"{save_dir}/best_classifier.pt"
    data_dir = "/mnt/data"
