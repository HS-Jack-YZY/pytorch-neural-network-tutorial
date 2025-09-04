import numpy as np
from sklearn.datasets import load_iris, make_classification, make_regression


def generate_regression_data(n_samples=1000, n_features=1, noise=0.1, random_state=42):
    np.random.seed(random_state)

    if n_features == 1:
        X = np.random.uniform(-3, 3, (n_samples, 1))
        y = (
            2 * X.ravel()
            + np.sin(2 * X.ravel())
            + np.random.normal(0, noise, n_samples)
        )
        y = y.reshape(-1, 1)
    else:
        X, y = make_regression(
            n_samples, n_features, noise=noise * 10, random_state=random_state
        )
        y = y.reshape(-1, 1)

    return X.astype(np.float32), y.astype(np.float32)


def generate_classification_data(
    n_samples=1000, n_features=2, n_classes=2, n_cluster=1, random_state=42
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=n_cluster,
        n_redundant=0,
        n_informative=n_features,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.float32)


def generate_iris_data():
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.float32)
    return X, y, iris.feature_names, iris.target_names


def generate_spiral_data(n_samples=200, n_classes=2, random_state=42):
    np.random.seed(random_state)
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype=np.int64)

    for class_idx in range(n_classes):
        theta = np.linspace(0, 4 * np.pi, n_samples) + class_idx * 4 * np.pi / n_classes
        radius = (
            theta / (4 * np.pi) + np.random.randn(n_samples) * 0.1
        )  # np.random.normal(0, 1, n_samples) * 0.1

        start_idx = class_idx * n_samples
        end_idx = (class_idx + 1) * n_samples

        X[start_idx:end_idx, 0] = radius * np.cos(theta)
        X[start_idx:end_idx, 1] = radius * np.sin(theta)
        y[start_idx:end_idx] = class_idx

    return X.astype(np.float32), y
