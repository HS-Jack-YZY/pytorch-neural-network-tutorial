"""
简单分类示例

这个脚本展示了如何使用我们构建的神经网络进行分类任务。
包含数据生成、模型训练、结果评估和可视化的完整流程。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from src.models import SimpleNeuralNetwork, MultiClassNetwork
from src.utils import train_model, evaluate_model, plot_decision_boundary, plot_cost_history
from src.data_loader import (
    create_spiral_data, create_moon_data, create_circles_data,
    preprocess_data, create_data_loaders, visualize_data, data_info
)


def main():
    """主函数：运行分类示例"""
    
    print("=== PyTorch神经网络分类示例 ===\n")
    
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 1. 生成不同类型的数据集
    datasets = {
        "螺旋数据": create_spiral_data(n_samples=300, n_classes=3),
        "月牙数据": create_moon_data(n_samples=600, noise=0.15),
        "同心圆数据": create_circles_data(n_samples=600, noise=0.1)
    }
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*50}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*50}")
        
        # 显示数据信息
        data_info(X, y, dataset_name)
        
        # 可视化原始数据
        visualize_data(X, y, f"原始{dataset_name}")
        
        # 2. 数据预处理
        X_train, X_test, y_train, y_test, scaler = preprocess_data(
            X, y, test_size=0.2, scale_features=True
        )
        
        # 创建数据加载器
        train_loader, test_loader = create_data_loaders(
            X_train, y_train, X_test, y_test, batch_size=32
        )
        
        # 3. 选择合适的模型
        n_classes = len(np.unique(y))
        
        if n_classes == 2:
            # 二分类问题：使用简单神经网络
            model = SimpleNeuralNetwork(
                input_dim=X.shape[1],
                hidden_dim=64,
                output_dim=1,
                task='classification'
            )
            criterion = nn.BCELoss()  # 二分类交叉熵损失
        else:
            # 多分类问题：使用多分类网络
            model = MultiClassNetwork(
                input_dim=X.shape[1],
                hidden_dims=[64, 32],
                num_classes=n_classes
            )
            criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
        
        # 4. 设置优化器
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        print(f"\n模型结构:")
        print(model)
        print(f"\n参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 5. 训练模型
        print(f"\n开始训练...")
        loss_history = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=200,
            device=device,
            verbose=True
        )
        
        # 6. 绘制训练损失曲线
        plot_cost_history(loss_history, f"{dataset_name} - 训练损失曲线")
        
        # 7. 评估模型
        print(f"\n评估模型性能...")
        results = evaluate_model(model, test_loader, device)
        
        if 'accuracy' in results:
            print(f"测试准确率: {results['accuracy']:.4f}")
            print(f"\n详细分类报告:")
            print(results['classification_report'])
        
        # 8. 可视化决策边界
        plot_decision_boundary(model, X_test, y_test, device)
        
        print(f"\n{dataset_name} 处理完成！")


def demonstrate_overfitting():
    """演示过拟合现象"""
    
    print(f"\n{'='*50}")
    print("过拟合演示")
    print(f"{'='*50}")
    
    # 生成小数据集
    X, y = create_moon_data(n_samples=100, noise=0.2)
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y, test_size=0.3)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建一个容量很大的模型
    model = SimpleNeuralNetwork(
        input_dim=2,
        hidden_dim=256,  # 非常大的隐藏层
        output_dim=1,
        task='classification'
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("训练大容量模型（容易过拟合）...")
    
    # 训练更多轮次以观察过拟合
    loss_history = train_model(
        model, train_loader, criterion, optimizer,
        num_epochs=500, device=device
    )
    
    # 评估并可视化
    results = evaluate_model(model, test_loader, device)
    print(f"测试准确率: {results['accuracy']:.4f}")
    
    plot_cost_history(loss_history, "大容量模型训练损失（观察过拟合）")
    plot_decision_boundary(model, X_test, y_test, device)


if __name__ == "__main__":
    # 运行主要分类示例
    main()
    
    # 演示过拟合
    demonstrate_overfitting()
    
    print(f"\n{'='*50}")
    print("所有示例运行完成！")
    print("现在您可以:")
    print("1. 尝试修改模型结构")
    print("2. 调整超参数")
    print("3. 使用不同的数据集")
    print("4. 实验不同的优化器和损失函数")
    print(f"{'='*50}")