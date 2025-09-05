config = {
    "seed": 42,
    "epochs": 2000,
    "batch_size": 128,
    "lr_G": 1e-5,
    "lr_D": 1e-5,
    "lambda_cls": 2.0,
    "lambda_reg": 1.0,
    "input_dim": 2,
    "hidden_dim": 32,
    "out_dir": "results",
    "clf_model_path": "results/classifier.pt",
    "cuda": "cuda" if __import__("torch").cuda.is_available() else "cpu",
}
