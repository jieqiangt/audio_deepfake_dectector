
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN_STFT_FRAMESIZE_1024(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 7, 2)
        self.conv2 = nn.Conv2d(96, 128, 5, 1)
        self.conv3 = nn.Conv2d(128, 256, 5, 1)
        self.conv4 = nn.Conv2d(256, 128, 5, 1)
        self.conv5 = nn.Conv2d(128, 64, 3, 1)
        self.conv6 = nn.Conv2d(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 8, 3, 1)
        self.fc1 = nn.Linear(26*2*8, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, X):
        # input size is (513, 127, 3)
        X = F.relu(self.conv1(X))  # output size (254, 61, 96)
        X = F.relu(self.conv2(X))  # output size (250, 57, 256)
        X = F.avg_pool2d(X, 2, 2)  # output size (125, 28, 256)
        X = F.dropout(X, 0.2)
        X = F.relu(self.conv3(X))  # output size (121, 24, 384)
        X = F.relu(self.conv4(X))  # output size (117, 20, 256)
        X = F.avg_pool2d(X, 2, 2)  # output size (58, 10, 256)
        X = F.dropout(X, 0.2)
        X = F.relu(self.conv5(X))  # output size (56, 8, 128)
        X = F.relu(self.conv6(X))  # output size (54, 6, 64)
        X = F.relu(self.conv7(X))  # output size (52, 4, 8)
        X = F.avg_pool2d(X, 2, 2)  # output size (26, 2, 8)
        X = F.dropout(X, 0.2)
        X = X.view(-1, 26*2*8)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.softmax(X, dim=1)
