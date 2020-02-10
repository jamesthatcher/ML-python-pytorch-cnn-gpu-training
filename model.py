import torch
from torch import nn


# Convolutional neural network
class CNN(nn.Module):

    def __init__(self, n_classes=10):
        super(CNN, self).__init__()

        # Convolutional layers
        self.layers = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    )

        # Fully connected layers with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(0.2),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
