"""
数据加载和预处理模块

提供各种数据生成和加载功能，包括合成数据生成、
真实数据集加载和数据预处理功能。
"""

import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def generate_regression_data(n_samples=1000, n_features=1, noise=0.1, random_state=42):
    """
    生成回归数据
    
    Args:
        n_samples (int): 样本数量
        n_features (int): 特征数量
        noise (float): 噪声水平
        random_state (int): 随机种子
        
    Returns:
        tuple: (X, y) 特征和标签
    """
    np.random.seed(random_state)
    
    if n_features == 1:
        # 一维数据：生成非线性关系
        X = np.random.uniform(-3, 3, (n_samples, 1))
        y = 2 * X.ravel() + np.sin(2 * X.ravel()) + np.random.normal(0, noise, n_samples)
        y = y.reshape(-1, 1)
    else:
        # 多维数据：使用sklearn生成
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise * 10,  # sklearn的noise参数范围不同
            random_state=random_state
        )
        y = y.reshape(-1, 1)
    
    return X.astype(np.float32), y.astype(np.float32)


def generate_classification_data(n_samples=1000, n_features=2, n_classes=2, 
                                n_clusters=1, random_state=42):
    """
    生成分类数据
    
    Args:
        n_samples (int): 样本数量
        n_features (int): 特征数量
        n_classes (int): 类别数量
        n_clusters (int): 每类的聚类数量
        random_state (int): 随机种子
        
    Returns:
        tuple: (X, y) 特征和标签
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters,
        n_redundant=0,
        n_informative=n_features,
        random_state=random_state
    )
    
    return X.astype(np.float32), y.astype(np.int64)


def load_iris_dataset():
    """
    加载鸢尾花数据集
    
    Returns:
        tuple: (X, y) 特征和标签，以及特征名和类别名
    """
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)
    
    return X, y, iris.feature_names, iris.target_names


def create_spiral_data(n_samples=1000, n_classes=3, random_state=42):
    """
    创建螺旋形分类数据
    
    这是一个经典的非线性分类问题，适合测试神经网络的非线性学习能力。
    
    Args:
        n_samples (int): 每个类别的样本数量
        n_classes (int): 类别数量
        random_state (int): 随机种子
        
    Returns:
        tuple: (X, y) 特征和标签
    """
    np.random.seed(random_state)
    
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype=np.int64)
    
    for class_idx in range(n_classes):
        # 生成螺旋的角度
        theta = np.linspace(0, 4 * np.pi, n_samples) + class_idx * 4 * np.pi / n_classes
        # 添加随机噪声
        radius = theta / (4 * np.pi) + np.random.randn(n_samples) * 0.1
        
        # 转换为笛卡尔坐标
        start_idx = class_idx * n_samples
        end_idx = (class_idx + 1) * n_samples
        
        X[start_idx:end_idx, 0] = radius * np.cos(theta)
        X[start_idx:end_idx, 1] = radius * np.sin(theta)
        y[start_idx:end_idx] = class_idx
    
    return X.astype(np.float32), y


def preprocess_data(X, y, test_size=0.2, random_state=42, scale_features=True):
    """
    数据预处理：分割训练测试集和特征标准化
    
    Args:
        X (np.array): 特征数据
        y (np.array): 标签数据
        test_size (float): 测试集比例
        random_state (int): 随机种子
        scale_features (bool): 是否标准化特征
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = None
    if scale_features:
        # 标准化特征
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32, shuffle=True):
    """
    创建PyTorch数据加载器
    
    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train) if y_train.dtype == np.float32 else torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test) if y_test.dtype == np.float32 else torch.LongTensor(y_test)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def visualize_data(X, y, title="数据可视化", feature_names=None):
    """
    可视化二维数据
    
    Args:
        X (np.array): 特征数据
        y (np.array): 标签数据
        title (str): 图表标题
        feature_names (list): 特征名称
    """
    if X.shape[1] != 2:
        print("只能可视化二维数据")
        return
    
    plt.figure(figsize=(10, 8))
    
    # 获取唯一的类别
    unique_classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    
    for i, class_label in enumerate(unique_classes):
        mask = y == class_label
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=[colors[i]], label=f'类别 {class_label}', 
                   alpha=0.7, s=50)
    
    plt.title(title, fontsize=16)
    plt.xlabel(feature_names[0] if feature_names else '特征 1', fontsize=12)
    plt.ylabel(feature_names[1] if feature_names else '特征 2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def create_moon_data(n_samples=1000, noise=0.1, random_state=42):
    """
    创建月牙形数据
    
    另一个经典的非线性分类问题。
    
    Args:
        n_samples (int): 样本数量
        noise (float): 噪声水平
        random_state (int): 随机种子
        
    Returns:
        tuple: (X, y) 特征和标签
    """
    from sklearn.datasets import make_moons
    
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X.astype(np.float32), y.astype(np.int64)


def create_circles_data(n_samples=1000, noise=0.1, factor=0.5, random_state=42):
    """
    创建同心圆数据
    
    用于测试模型处理复杂边界的能力。
    
    Args:
        n_samples (int): 样本数量
        noise (float): 噪声水平
        factor (float): 内外圆的比例
        random_state (int): 随机种子
        
    Returns:
        tuple: (X, y) 特征和标签
    """
    from sklearn.datasets import make_circles
    
    X, y = make_circles(n_samples=n_samples, noise=noise, 
                       factor=factor, random_state=random_state)
    return X.astype(np.float32), y.astype(np.int64)


# 数据统计信息函数
def data_info(X, y, dataset_name="数据集"):
    """
    打印数据集的基本信息
    
    Args:
        X (np.array): 特征数据
        y (np.array): 标签数据
        dataset_name (str): 数据集名称
    """
    print(f"\n=== {dataset_name} 信息 ===")
    print(f"样本数量: {X.shape[0]}")
    print(f"特征数量: {X.shape[1]}")
    print(f"标签形状: {y.shape}")
    
    if len(np.unique(y)) < 20:  # 分类问题
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"类别数量: {len(unique_labels)}")
        print("类别分布:")
        for label, count in zip(unique_labels, counts):
            print(f"  类别 {label}: {count} 个样本")
    else:  # 回归问题
        print(f"标签范围: [{y.min():.3f}, {y.max():.3f}]")
        print(f"标签均值: {y.mean():.3f}")
        print(f"标签标准差: {y.std():.3f}")
    
    print(f"特征统计:")
    print(f"  最小值: {X.min(axis=0)}")
    print(f"  最大值: {X.max(axis=0)}")
    print(f"  均值: {X.mean(axis=0)}")
    print()