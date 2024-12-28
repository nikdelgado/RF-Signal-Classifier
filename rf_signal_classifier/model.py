import torch.nn as nn

class RFSignalClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RFSignalClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),  # Batch normalization after first convolution
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),  # Batch normalization after second convolution
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),  # Batch normalization after third convolution
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(128 * (input_size // 8), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)