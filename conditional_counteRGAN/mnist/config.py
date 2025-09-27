import torch

class Config:
    batch_size = 128
    num_workers = 4

    num_epochs_gan = 25
    num_epochs_clf = 10
    cls_lr = 1e-3
    d_lr = 1e-5
    g_lr = 5e-5
    lambda_adv = 1.0
    lambda_cls = 1.0
    lambda_reg = 2.5

    latent_dim = 100
    img_shape = (1, 28, 28)
    num_classes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./results"
    classifier_path = f"{save_dir}/best_classifier.pt"
    data_dir = "/mnt/data"
