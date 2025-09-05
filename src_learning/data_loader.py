import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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


def preprocess_data(X, y, test_size=0.2, random_state=42, scale_features=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 是否需要标准化
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=64, shuffle=True):
    # 转化为张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = (
        torch.FloatTensor(y_train)
        if y_train.dtype == np.float32
        else torch.LongTensor(y_train)
    )
    y_test_tensor = (
        torch.FloatTensor(y_test)
        if y_test.dtype == np.float32
        else torch.LongTensor(y_test)
    )

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def visualize_data(X, y, title="数据可视化", feature_names=None):
    # 检测是否为二维图表
    if X.shape[1] != 2:
        print("只能可视化二维图标")
        return

    plt.figure(figsize=(10, 8))
    # 统计类别数量
    unique_classes = np.unique(y)
    # 颜色映射
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

    # 开始画散点图
    for i, class_label in enumerate(unique_classes):
        # 将同一类点全部找出来
        mask = y == class_label
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            c=colors[i],
            alpha=0.7,
            s=50,
            label=f"类别 {class_label}",
        )

    plt.title(title, fontsize=16)
    plt.xlabel(feature_names[0] if feature_names else "特征 1", fontsize=12)
    plt.ylabel(feature_names[1] if feature_names else "特征 2", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
