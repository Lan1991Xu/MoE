#multi-layer perceptron
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(64, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def mlp(**kwargs):
    return MLP(**kwargs)
