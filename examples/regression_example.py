"""
回归示例

展示如何使用PyTorch神经网络解决回归问题。
包含线性回归和非线性回归的完整示例。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from src.models import LinearRegression, SimpleNeuralNetwork, DeepNeuralNetwork
from src.utils import train_model, evaluate_model, plot_cost_history, plot_regression_results
from src.data_loader import generate_regression_data, preprocess_data, create_data_loaders, data_info


def linear_regression_example():
    """线性回归示例"""
    
    print(f"\n{'='*50}")
    print("线性回归示例")
    print(f"{'='*50}")
    
    # 生成线性数据
    np.random.seed(42)
    X = np.random.uniform(-2, 2, (200, 1))
    y = 3 * X + 2 + np.random.normal(0, 0.5, X.shape)
    
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    data_info(X, y, "线性回归数据")
    
    # 数据预处理
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y, scale_features=False)
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)
    
    # 创建线性回归模型
    model = LinearRegression(input_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    print(f"\n模型结构: {model}")
    
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_history = train_model(model, train_loader, criterion, optimizer, 
                              num_epochs=100, device=device)
    
    # 评估模型
    results = evaluate_model(model, test_loader, device)
    print(f"\n测试结果:")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    
    # 获取学到的参数
    with torch.no_grad():
        weight = model.linear.weight.item()
        bias = model.linear.bias.item()
        print(f"\n学到的参数:")
        print(f"权重 (斜率): {weight:.4f} (真实值: 3.0)")
        print(f"偏置 (截距): {bias:.4f} (真实值: 2.0)")
    
    # 可视化结果
    plot_cost_history(loss_history, "线性回归训练损失")
    plot_regression_results(model, X_test, y_test, device)


def nonlinear_regression_example():
    """非线性回归示例"""
    
    print(f"\n{'='*50}")
    print("非线性回归示例")
    print(f"{'='*50}")
    
    # 生成非线性数据
    X, y = generate_regression_data(n_samples=800, n_features=1, noise=0.2)
    
    data_info(X, y, "非线性回归数据")
    
    # 可视化原始数据
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6)
    plt.title("非线性回归数据")
    plt.xlabel("输入特征")
    plt.ylabel("目标值")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 比较不同模型
    models = {
        "线性模型": LinearRegression(input_dim=1),
        "简单神经网络": SimpleNeuralNetwork(input_dim=1, hidden_dim=32, output_dim=1),
        "深度神经网络": DeepNeuralNetwork(input_dim=1, hidden_dims=[64, 32, 16], output_dim=1)
    }
    
    results_comparison = {}
    
    for model_name, model in models.items():
        print(f"\n训练 {model_name}...")
        
        # 设置优化器
        if "线性" in model_name:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        criterion = nn.MSELoss()
        
        # 训练模型
        loss_history = train_model(
            model, train_loader, criterion, optimizer,
            num_epochs=200, device=device, verbose=False
        )
        
        # 评估模型
        test_results = evaluate_model(model, test_loader, device)
        results_comparison[model_name] = test_results
        
        print(f"{model_name} - 测试RMSE: {test_results['rmse']:.4f}")
        
        # 可视化训练过程
        plot_cost_history(loss_history, f"{model_name} - 训练损失")
        
        # 可视化回归结果
        plot_regression_results(model, X_test, y_test, device)
    
    # 结果比较
    print(f"\n{'='*50}")
    print("模型性能比较")
    print(f"{'='*50}")
    
    for model_name, results in results_comparison.items():
        print(f"{model_name:15} - RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}")


def multidimensional_regression():
    """多维回归示例"""
    
    print(f"\n{'='*50}")
    print("多维回归示例")
    print(f"{'='*50}")
    
    # 生成多维数据
    X, y = generate_regression_data(n_samples=1000, n_features=5, noise=0.1)
    
    data_info(X, y, "多维回归数据")
    
    # 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)
    
    # 创建深度神经网络
    model = DeepNeuralNetwork(
        input_dim=5,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        task='regression'
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print(f"\n模型结构:")
    print(model)
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_history = train_model(
        model, train_loader, criterion, optimizer,
        num_epochs=300, device=device
    )
    
    # 评估模型
    results = evaluate_model(model, test_loader, device)
    print(f"\n最终测试结果:")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    
    # 可视化结果
    plot_cost_history(loss_history, "多维回归训练损失")
    plot_regression_results(model, X_test, y_test, device)


def regularization_comparison():
    """正则化效果比较"""
    
    print(f"\n{'='*50}")
    print("正则化效果比较")
    print(f"{'='*50}")
    
    # 生成小数据集（容易过拟合）
    X, y = generate_regression_data(n_samples=100, n_features=1, noise=0.1)
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y)
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 比较不同正则化程度
    weight_decays = [0, 1e-4, 1e-3, 1e-2]
    
    for wd in weight_decays:
        print(f"\n权重衰减: {wd}")
        
        model = SimpleNeuralNetwork(
            input_dim=1,
            hidden_dim=128,  # 大容量模型
            output_dim=1
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=wd)
        
        # 训练模型
        loss_history = train_model(
            model, train_loader, criterion, optimizer,
            num_epochs=500, device=device, verbose=False
        )
        
        # 评估
        results = evaluate_model(model, test_loader, device)
        print(f"测试RMSE: {results['rmse']:.4f}")
        
        # 可视化
        plot_cost_history(loss_history, f"权重衰减={wd} - 训练损失")


def main():
    """主函数"""
    
    print("=== PyTorch回归示例教程 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行各种回归示例
    linear_regression_example()
    nonlinear_regression_example()
    multidimensional_regression()
    regularization_comparison()
    
    print(f"\n{'='*50}")
    print("所有回归示例完成！")
    print("您学到了:")
    print("1. 线性回归的PyTorch实现")
    print("2. 神经网络处理非线性关系")
    print("3. 多维特征的回归")
    print("4. 正则化的重要性")
    print("5. 不同模型的性能比较")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()