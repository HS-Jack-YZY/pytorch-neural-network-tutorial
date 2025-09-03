import numpy as np
from sklearn.datasets import make_regression


def generate_regression_data(n_numbers=1000, n_features=1, noise=0.1, random_state=42):
    np.random.seed(random_state)

    if n_features == 1:
        X = np.random.uniform(-3, 3, (n_numbers, 1))
        y = (
            2 * X.ravel()
            + np.sin(2 * X.ravel())
            + np.random.normal(0, noise, n_numbers)
        )
        y = y.reshape(-1, 1)
    else:
        X, y = make_regression(
            n_numbers, n_features, noise=noise * 10, random_state=random_state
        )
        y = y.reshap(-1, 1)

    return (X, y)
