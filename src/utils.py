"""
工具函数模块

包含了训练、评估、可视化等常用工具函数。
这些函数可以帮助您更方便地进行模型开发和结果分析。
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def train_model(model, train_loader, criterion, optimizer, num_epochs=100, device='cpu', verbose=True):
    """
    通用模型训练函数
    
    Args:
        model: PyTorch模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs (int): 训练轮数
        device (str): 设备类型 ('cpu' 或 'cuda')
        verbose (bool): 是否显示训练进度
        
    Returns:
        list: 损失历史记录
    """
    model.to(device)
    model.train()
    
    loss_history = []
    
    # 使用tqdm显示训练进度
    epoch_iterator = tqdm(range(num_epochs), desc="训练进度") if verbose else range(num_epochs)
    
    for epoch in epoch_iterator:
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # 更新进度条信息
        if verbose and epoch % 10 == 0:
            epoch_iterator.set_postfix({'损失': f'{avg_loss:.4f}'})
    
    return loss_history


def evaluate_model(model, test_loader, device='cpu'):
    """
    模型评估函数
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device (str): 设备类型
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            
            # 收集预测结果和真实标签
            if outputs.shape[1] == 1:  # 二分类或回归
                predictions = (outputs > 0.5).float() if hasattr(model, 'task') and model.task == 'classification' else outputs
            else:  # 多分类
                predictions = torch.argmax(outputs, dim=1)
                
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # 转换为numpy数组
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # 计算评估指标
    results = {}
    
    # 分类任务
    if hasattr(model, 'task') and model.task == 'classification':
        results['accuracy'] = accuracy_score(targets, predictions)
        results['classification_report'] = classification_report(targets, predictions)
        results['confusion_matrix'] = confusion_matrix(targets, predictions)
    else:  # 回归任务
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        results['mse'] = mse
        results['rmse'] = rmse
        results['mae'] = mae
    
    return results


def plot_cost_history(loss_history, title="训练损失曲线"):
    """
    绘制损失函数历史曲线
    
    Args:
        loss_history (list): 损失历史记录
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel('训练轮数 (Epoch)', fontsize=12)
    plt.ylabel('损失值 (Loss)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_decision_boundary(model, X, y, device='cpu', resolution=100):
    """
    绘制二维数据的决策边界
    
    Args:
        model: 训练好的模型
        X (np.array): 输入数据 (n_samples, 2)
        y (np.array): 标签数据
        device (str): 设备类型
        resolution (int): 网格分辨率
    """
    model.eval()
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid_points).to(device)
    
    with torch.no_grad():
        predictions = model(grid_tensor)
        if predictions.shape[1] == 1:  # 二分类
            Z = predictions.cpu().numpy().reshape(xx.shape)
        else:  # 多分类
            Z = torch.argmax(predictions, dim=1).cpu().numpy().reshape(xx.shape)
    
    # 绘制决策边界
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # 绘制数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.title('决策边界可视化', fontsize=16)
    plt.xlabel('特征 1', fontsize=12)
    plt.ylabel('特征 2', fontsize=12)
    plt.show()


def plot_regression_results(model, X, y, device='cpu'):
    """
    绘制回归结果
    
    Args:
        model: 训练好的回归模型
        X (np.array): 输入数据
        y (np.array): 真实标签
        device (str): 设备类型
    """
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
    plt.figure(figsize=(12, 5))
    
    # 如果是一维输入，绘制拟合曲线
    if X.shape[1] == 1:
        # 排序以便绘制平滑曲线
        sorted_indices = np.argsort(X[:, 0])
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        pred_sorted = predictions[sorted_indices]
        
        plt.subplot(1, 2, 1)
        plt.scatter(X, y, alpha=0.6, label='真实数据')
        plt.plot(X_sorted, pred_sorted, 'r-', linewidth=2, label='预测曲线')
        plt.xlabel('输入特征')
        plt.ylabel('目标值')
        plt.title('回归拟合结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 绘制预测值vs真实值散点图
    plt.subplot(1, 2, 2) if X.shape[1] == 1 else plt.subplot(1, 1, 1)
    plt.scatter(y, predictions, alpha=0.6)
    
    # 绘制完美预测线
    min_val = min(y.min(), predictions.min())
    max_val = max(y.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
    
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 真实值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names=None):
    """
    绘制混淆矩阵
    
    Args:
        cm (np.array): 混淆矩阵
        class_names (list): 类别名称列表
    """
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f'类别 {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵', fontsize=16)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.show()


def save_model(model, filepath, metadata=None):
    """
    保存模型
    
    Args:
        model: PyTorch模型
        filepath (str): 保存路径
        metadata (dict): 额外的元数据
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if metadata:
        save_dict.update(metadata)
    
    torch.save(save_dict, filepath)
    print(f"模型已保存到: {filepath}")


def load_model(model_class, filepath, **kwargs):
    """
    加载模型
    
    Args:
        model_class: 模型类
        filepath (str): 模型文件路径
        **kwargs: 模型初始化参数
        
    Returns:
        加载的模型
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model = model_class(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"模型已从 {filepath} 加载")
    return model