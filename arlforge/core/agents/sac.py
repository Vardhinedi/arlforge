import torch
import torch.nn.functional as F
from torch.optim import Adam
from arlforge.utils.networks import MLP


class SACAgent:
    """
    A simplified Soft Actor-Critic agent.
    Works for both continuous and discrete actions.
    """

    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.action_dim = action_dim  # <-- needed for one-hot

        # ---------------------------------------------------------
        # Networks
        # ---------------------------------------------------------
        self.actor = MLP(state_dim, action_dim)  # logits or actions

        # Q networks
        self.q1 = MLP(state_dim + action_dim, 1)
        self.q2 = MLP(state_dim + action_dim, 1)

        # Target Q networks
        self.q1_target = MLP(state_dim + action_dim, 1)
        self.q2_target = MLP(state_dim + action_dim, 1)

        # Copy weights
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=lr)

        self.tau = 0.005  # soft update factor

    # ---------------------------------------------------------
    # SELECT ACTION
    # ---------------------------------------------------------
    def select_action(self, state):
        """Continuous action: actor outputs tanh-scaled vector."""
        with torch.no_grad():
            action = self.actor(state)
            return torch.tanh(action)

    # ---------------------------------------------------------
    # UPDATE SAC
    # ---------------------------------------------------------
    def update(self, buffer, batch_size=64):
        if len(buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        # Convert states
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

        # ------------------------------------------------------
        # FIX: HANDLE DISCRETE ACTIONS (convert to one-hot)
        # ------------------------------------------------------
        actions = torch.tensor(actions)

        # Case 1 — DISCRETE: actions are integers → shape [batch]
        if actions.dim() == 1:
            actions = F.one_hot(actions.long(), num_classes=self.action_dim).float()

        # Case 2 — CONTINUOUS: already vectors → convert to float
        else:
            actions = actions.float()

        # ------------------------------------------------------
        # TARGET Q VALUES
        # ------------------------------------------------------
        with torch.no_grad():
            next_actions = self.select_action(next_states)
            next_input = torch.cat([next_states, next_actions], dim=-1)

            q1_t = self.q1_target(next_input)
            q2_t = self.q2_target(next_input)

            target_q = rewards + self.gamma * (1 - dones) * torch.min(q1_t, q2_t)

        # ------------------------------------------------------
        # UPDATE Q1
        # ------------------------------------------------------
        q1_input = torch.cat([states, actions], dim=-1)
        q1_val = self.q1(q1_input)
        q1_loss = F.mse_loss(q1_val, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # ------------------------------------------------------
        # UPDATE Q2
        # ------------------------------------------------------
        q2_input = torch.cat([states, actions], dim=-1)
        q2_val = self.q2(q2_input)
        q2_loss = F.mse_loss(q2_val, target_q)

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # ------------------------------------------------------
        # UPDATE ACTOR
        # ------------------------------------------------------
        new_actions = self.select_action(states)
        actor_input = torch.cat([states, new_actions], dim=-1)

        q1_new = self.q1(actor_input)
        actor_loss = -q1_new.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------------------------
        # SOFT UPDATE TARGET NETWORKS
        # ------------------------------------------------------
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item()
        }

    # ---------------------------------------------------------
    # SOFT UPDATE FUNCTION
    # ---------------------------------------------------------
    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
