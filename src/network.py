import string
import torch.nn as nn
import torch.nn.functional as F


class CaptchaDenseNetwork(nn.Module):
    def __init__(self):
        super(CaptchaDenseNetwork, self).__init__()
        number_classes = len(string.printable)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=16 * 11 * 48, out_features=1024)
        self.fc2_0 = nn.Linear(in_features=1024, out_features=number_classes)
        self.fc2_1 = nn.Linear(in_features=1024, out_features=number_classes)
        self.fc2_2 = nn.Linear(in_features=1024, out_features=number_classes)
        self.fc2_3 = nn.Linear(in_features=1024, out_features=number_classes)
        self.fc2_4 = nn.Linear(in_features=1024, out_features=number_classes)

    def forward(self, x):
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool2(F.relu(self.conv2(out)))
        # Flatten.
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        y0 = self.fc2_0(out)
        y1 = self.fc2_1(out)
        y2 = self.fc2_2(out)
        y3 = self.fc2_3(out)
        y4 = self.fc2_4(out)
        return y0, y1, y2, y3, y4
