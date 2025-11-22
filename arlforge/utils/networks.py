import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple feed-forward neural network used in many RL components.
    """

    def __init__(self, input_dim, output_dim, hidden_sizes=[64, 64]):
        super().__init__()

        layers = []
        in_features = input_dim

        # Hidden layers
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h

        # Output layer
        layers.append(nn.Linear(in_features, output_dim))

        # Register the network
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
