import torch
import torch.nn as nn
from arlforge.utils.networks import MLP

class DynamicsModel(nn.Module):
    """
    Predicts:
      - next_state
      - reward

    This is the core model used for synthetic rollouts (model-based RL).
    """

    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64]):
        super().__init__()

        # One combined model: input = [state, action]
        self.model = MLP(
            input_dim = state_dim + action_dim,
            output_dim = state_dim + 1,   # next_state dims + reward
            hidden_sizes = hidden_sizes
        )

    def forward(self, state, action):
        # state: [batch, state_dim]
        # action: [batch, action_dim]
        x = torch.cat([state, action], dim=-1)

        out = self.model(x)

        next_state = out[..., :-1]
        reward     = out[..., -1:]

        return next_state, reward

    def train_step(self, optimizer, loss_fn, state, action, next_state_target, reward_target):
        """
        One training step on a batch of data.
        """
        optimizer.zero_grad()

        pred_next, pred_reward = self.forward(state, action)

        loss_state = loss_fn(pred_next, next_state_target)
        loss_reward = loss_fn(pred_reward, reward_target)

        loss = loss_state + loss_reward
        loss.backward()
        optimizer.step()

        return loss.item()
