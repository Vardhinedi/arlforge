import torch
from arlforge.utils.networks import MLP

net = MLP(input_dim=3, output_dim=1)

x = torch.tensor([[1.0, 2.0, 3.0]])
y = net(x)

print("Input:", x)
print("Output:", y)
