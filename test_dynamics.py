import torch
from arlforge.core.dynamics_model import DynamicsModel
from torch.optim import Adam
import torch.nn as nn

state_dim = 3
action_dim = 1

model = DynamicsModel(state_dim, action_dim)
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Fake data
state = torch.tensor([[1.0, 2.0, 3.0]])
action = torch.tensor([[0.5]])
next_state_target = torch.tensor([[1.1, 2.2, 3.3]])
reward_target = torch.tensor([[0.8]])

# Train once
loss = model.train_step(optimizer, loss_fn, state, action, next_state_target, reward_target)
print("Loss:", loss)

# Predict
pred_next, pred_reward = model(state, action)
print("Predicted next:", pred_next)
print("Predicted reward:", pred_reward)
