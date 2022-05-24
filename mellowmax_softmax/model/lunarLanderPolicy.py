import torch.nn as nn
import torch.nn.functional as F


class policy(nn.Module):

    def __init__(self, inputSize=8, outputSize=4, hidden_size=16):
        super(policy, self).__init__()
        self.layer1 = nn.Linear(inputSize, hidden_size)
        self.layer2 = nn.Linear(hidden_size, outputSize)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = self.layer2(x)
        return x