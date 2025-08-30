import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(seed=42):
    np.random.seed(seed)
    X_moons, y_moons = make_moons(n_samples=800, noise=0.1)

    # additional rectangle class
    X_rect = np.random.uniform(low=[-2, 2], high=[2, 4], size=(400, 2))
    y_rect = np.full(400, 2)

    X = np.vstack([X_moons, X_rect])
    y = np.concatenate([y_moons, y_rect])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    return X_train, X_test, y_train, y_test