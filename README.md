# PyTorch神经网络教程

这是一个完整的PyTorch神经网络教程项目，专为学习机器学习的同学设计，可以作为吴恩达机器学习课程的PyTorch替代版本。

## 🎯 项目目标

本教程将帮助您：
- 掌握PyTorch的基础操作
- 理解神经网络的核心概念
- 实现线性回归、逻辑回归和深度神经网络
- 学会使用PyTorch进行模型训练和评估
- 掌握深度学习的最佳实践

## 📁 项目结构

```
pytorch-neural-network-tutorial/
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖包列表
├── notebooks/                   # Jupyter教程笔记本
│   ├── 01_pytorch_basics.ipynb      # PyTorch基础
│   ├── 02_linear_regression.ipynb   # 线性回归
│   ├── 03_logistic_regression.ipynb # 逻辑回归
│   ├── 04_neural_network_basics.ipynb # 神经网络基础
│   └── 05_deep_neural_network.ipynb   # 深度神经网络
├── src/                         # 核心代码模块
│   ├── __init__.py
│   ├── models.py               # 模型定义
│   ├── utils.py                # 工具函数
│   └── data_loader.py          # 数据加载器
└── examples/                    # 示例脚本
    ├── simple_classification.py    # 分类示例
    └── regression_example.py       # 回归示例
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone https://github.com/your-username/pytorch-neural-network-tutorial.git
cd pytorch-neural-network-tutorial

# 安装依赖
pip install -r requirements.txt

# 启动Jupyter
jupyter notebook
```

### 2. 学习路径

建议按以下顺序学习：

1. **PyTorch基础** (`notebooks/01_pytorch_basics.ipynb`)
   - 张量操作
   - 自动求导
   - GPU加速

2. **线性回归** (`notebooks/02_linear_regression.ipynb`)
   - 最简单的机器学习模型
   - 梯度下降优化

3. **逻辑回归** (`notebooks/03_logistic_regression.ipynb`)
   - 分类问题入门
   - Sigmoid激活函数

4. **神经网络基础** (`notebooks/04_neural_network_basics.ipynb`)
   - 多层感知机
   - 激活函数和损失函数

5. **深度神经网络** (`notebooks/05_deep_neural_network.ipynb`)
   - 深层网络
   - 正则化技术

## 🛠 核心功能

### 模型实现
- ✅ 线性回归模型
- ✅ 逻辑回归模型  
- ✅ 多层感知机
- ✅ 深度神经网络

### 工具函数
- ✅ 数据可视化
- ✅ 模型训练和评估
- ✅ 性能监控
- ✅ 结果分析

### 示例项目
- ✅ 房价预测（回归）
- ✅ 鸢尾花分类
- ✅ 手写数字识别

## 📖 主要特性

- **中文注释**：所有代码都有详细的中文注释
- **渐进式学习**：从简单到复杂，循序渐进
- **可视化丰富**：大量图表帮助理解
- **实践导向**：理论与实践相结合
- **GPU支持**：支持CUDA加速训练

## 🔧 环境要求

- Python 3.8+
- PyTorch 2.0+
- Jupyter Notebook
- CUDA（可选，用于GPU加速）

## 📚 参考资料

- [PyTorch官方文档](https://pytorch.org/docs/)
- [吴恩达机器学习课程](https://www.coursera.org/learn/machine-learning)
- [深度学习花书](https://www.deeplearningbook.org/)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个教程！

## 📄 许可证

MIT License

---

**开始学习之旅吧！🚀**