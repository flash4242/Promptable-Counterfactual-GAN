config = {
    "seed": 42,
    "epochs": 600,
    "batch_size": 128,
    "lr_G": 3e-4,
    "lr_D": 1e-4,
    "lambda_cls": 2.0,
    "lambda_reg": 3.0,
    "input_dim": 2,
    "hidden_dim": 32,
    "out_dir": "results",
    "clf_model_path": "results/classifier.pt",
    "cuda": "cuda" if __import__("torch").cuda.is_available() else "cpu",
}