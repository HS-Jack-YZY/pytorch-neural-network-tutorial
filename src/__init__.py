"""
PyTorch神经网络教程 - 核心模块

这个包包含了机器学习和深度学习的核心实现，
包括模型定义、工具函数和数据加载器。
"""

__version__ = "1.0.0"
__author__ = "PyTorch Tutorial Team"

from .models import LinearRegression, LogisticRegression, SimpleNeuralNetwork, DeepNeuralNetwork
from .utils import plot_decision_boundary, plot_cost_history, train_model, evaluate_model
from .data_loader import generate_classification_data, generate_regression_data

__all__ = [
    'LinearRegression',
    'LogisticRegression', 
    'SimpleNeuralNetwork',
    'DeepNeuralNetwork',
    'plot_decision_boundary',
    'plot_cost_history',
    'train_model',
    'evaluate_model',
    'generate_classification_data',
    'generate_regression_data'
]