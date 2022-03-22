from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        n_classes = 2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3))
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3))
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,3))
        self.pool4 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)

        return x

