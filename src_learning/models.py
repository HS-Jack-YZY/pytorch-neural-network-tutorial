import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegression(nn.Module):
    def __init__(self, input_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, task="regression"):
        super(SimpleNeuralNetwork, self).__init__()
        self.task = task
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.task == "classification":
            x = torch.sigmoid(self.fc2(x))
        else:
            x = self.fc2(x)
        return x


class DeepNeuralNetwork(nn.Module):
    def __init__(
        self, input_dim=1, hidden_dims=[128, 64], output_dim=1, task="regression"
    ):
        super(DeepNeuralNetwork, self).__init__()
        self.task = task
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        if self.task == "classification":
            x = torch.sigmoid(self.output_layer(x))
        else:
            x = self.output_layer(x)
        return x
