import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes, conv_channels=None, fc_size=256, dropout=0.3):
        super(CNN, self).__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.conv_layers = nn.ModuleList()
        in_channels = 1

        for out_channels in conv_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
            in_channels = out_channels

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(conv_channels[-1], fc_size)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x
