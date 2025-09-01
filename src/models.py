"""
模型定义模块

包含了线性回归、逻辑回归、简单神经网络和深度神经网络的实现。
每个模型都继承自nn.Module，包含标准的前向传播方法。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegression(nn.Module):
    """
    线性回归模型
    
    这是最简单的机器学习模型，用于预测连续值。
    模型形式：y = wx + b
    """
    
    def __init__(self, input_dim=1):
        """
        初始化线性回归模型
        
        Args:
            input_dim (int): 输入特征的维度，默认为1
        """
        super(LinearRegression, self).__init__()
        # 定义线性层：输入维度 -> 1个输出
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入数据，形状为 (batch_size, input_dim)
            
        Returns:
            torch.Tensor: 预测结果，形状为 (batch_size, 1)
        """
        return self.linear(x)


class LogisticRegression(nn.Module):
    """
    逻辑回归模型
    
    用于二分类问题，输出概率值在0到1之间。
    使用Sigmoid激活函数将线性输出转换为概率。
    """
    
    def __init__(self, input_dim):
        """
        初始化逻辑回归模型
        
        Args:
            input_dim (int): 输入特征的维度
        """
        super(LogisticRegression, self).__init__()
        # 线性层：特征维度 -> 1个输出
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入数据
            
        Returns:
            torch.Tensor: 经过Sigmoid激活的概率输出
        """
        # 线性变换后应用Sigmoid激活函数
        return torch.sigmoid(self.linear(x))


class SimpleNeuralNetwork(nn.Module):
    """
    简单的神经网络模型
    
    包含一个隐藏层的多层感知机(MLP)，
    适用于非线性分类和回归问题。
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, task='regression'):
        """
        初始化简单神经网络
        
        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层神经元数量，默认64
            output_dim (int): 输出维度，默认1
            task (str): 任务类型，'regression' 或 'classification'
        """
        super(SimpleNeuralNetwork, self).__init__()
        self.task = task
        
        # 第一层：输入 -> 隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 第二层：隐藏层 -> 输出
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Dropout层用于正则化，防止过拟合
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入数据
            
        Returns:
            torch.Tensor: 模型输出
        """
        # 第一层 + ReLU激活 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 输出层
        x = self.fc2(x)
        
        # 根据任务类型选择激活函数
        if self.task == 'classification':
            x = torch.sigmoid(x)  # 分类任务使用Sigmoid
            
        return x


class DeepNeuralNetwork(nn.Module):
    """
    深度神经网络模型
    
    包含多个隐藏层的深度网络，
    具有更强的表达能力，适合复杂的学习任务。
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1, task='regression'):
        """
        初始化深度神经网络
        
        Args:
            input_dim (int): 输入特征维度
            hidden_dims (list): 每个隐藏层的神经元数量列表
            output_dim (int): 输出维度
            task (str): 任务类型
        """
        super(DeepNeuralNetwork, self).__init__()
        self.task = task
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),  # 批归一化
                nn.Dropout(0.3)              # Dropout正则化
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 将所有层组合成Sequential模型
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入数据
            
        Returns:
            torch.Tensor: 模型输出
        """
        x = self.network(x)
        
        # 根据任务类型选择最后的激活函数
        if self.task == 'classification':
            x = torch.sigmoid(x)
            
        return x


# 多分类神经网络
class MultiClassNetwork(nn.Module):
    """
    多分类神经网络
    
    专门用于多分类问题，输出每个类别的概率分布。
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=3):
        """
        初始化多分类网络
        
        Args:
            input_dim (int): 输入特征维度
            hidden_dims (list): 隐藏层维度列表
            num_classes (int): 类别数量
        """
        super(MultiClassNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        # 输出层：输出每个类别的得分
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入数据
            
        Returns:
            torch.Tensor: 类别概率分布
        """
        logits = self.network(x)
        # 使用Softmax将得分转换为概率分布
        return F.softmax(logits, dim=1)