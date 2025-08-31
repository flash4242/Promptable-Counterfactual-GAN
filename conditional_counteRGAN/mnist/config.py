import torch

class Config:
    batch_size = 128
    num_workers = 4

    num_epochs_gan = 10
    num_epochs_clf = 10
    lr = 2e-4
    lambda_cls = 1.0
    lambda_reg = 0.5
    lambda_cyc = 2.0
    p_id = 0.1    # identity minibatch fraction



    latent_dim = 100
    img_shape = (1, 28, 28)
    num_classes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./results"
    classifier_path = f"{save_dir}/best_classifier.pt"
    data_dir = "/mnt/data"
