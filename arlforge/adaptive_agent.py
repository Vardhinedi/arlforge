import torch
import torch.nn.functional as F

from arlforge.utils.replay_buffer import ReplayBuffer
from arlforge.core.dynamics_model import DynamicsModel
from arlforge.core.agents.sac import SACAgent
from arlforge.utils.normalizer import Normalizer


class AdaptiveAgent:
    """
    Adaptive RL agent combining:
    - Real environment experience
    - Learned dynamics model
    - Synthetic rollouts
    - SAC trained on both
    """

    def __init__(self, env, config=None):
        self.env = env
        self.config = config or {}

        # ---------------------------
        # Observation space
        # ---------------------------
        self.obs_dim = env.observation_space.shape[0]

        # State normalizer
        self.normalizer = Normalizer(self.obs_dim)

        # ---------------------------
        # Action space
        # ---------------------------
        if env.action_space.__class__.__name__ == "Discrete":
            self.is_discrete = True
            self.act_dim = env.action_space.n
        else:
            self.is_discrete = False
            self.act_dim = env.action_space.shape[0]

        # ---------------------------
        # Replay buffers
        # ---------------------------
        self.real_buffer = ReplayBuffer(capacity=100000)
        self.model_buffer = ReplayBuffer(capacity=100000)

        # ---------------------------
        # SAC agent
        # ---------------------------
        self.agent = SACAgent(self.obs_dim, self.act_dim)

        # ---------------------------
        # Dynamics model
        # ---------------------------
        dyn_action_dim = self.act_dim
        self.model = DynamicsModel(self.obs_dim, dyn_action_dim)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    # ============================================================
    # Dynamics model training
    # ============================================================
    def train_dynamics_model(self, batch_size=64):
        if len(self.real_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.real_buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # Discrete action = convert to one-hot
        if self.is_discrete:
            actions = torch.tensor(actions, dtype=torch.long)
            actions = F.one_hot(actions, num_classes=self.act_dim).float()
        else:
            actions = torch.tensor(actions, dtype=torch.float32)

        loss = self.model.train_step(
            self.model_optimizer,
            torch.nn.MSELoss(),
            states,
            actions,
            next_states,
            rewards,
        )

        return loss

    # ============================================================
    # Synthetic model rollouts
    # ============================================================
    def generate_model_rollouts(self, num_samples=20):
        if len(self.real_buffer) == 0:
            return

        sampled_state, _, _, _, _ = self.real_buffer.sample(1)
        state = torch.tensor([sampled_state[0]], dtype=torch.float32)

        for _ in range(num_samples):

            if self.is_discrete:
                logits = self.agent.actor(state)
                action_idx = torch.argmax(logits, dim=1).item()
                action_tensor = F.one_hot(
                    torch.tensor([action_idx]),
                    num_classes=self.act_dim
                ).float()
            else:
                action_tensor = self.agent.select_action(state).detach()

            next_state_pred, reward_pred = self.model(state, action_tensor)

            self.model_buffer.add(
                state.squeeze().tolist(),
                action_tensor.squeeze().tolist(),
                reward_pred.item(),
                next_state_pred.squeeze().tolist(),
                False
            )

            state = next_state_pred

    # ============================================================
    # Main training loop
    # ============================================================
    def train(self, total_steps=300, model_rollouts_per_step=5):

        # Initial reset
        state, _ = self.env.reset()
        self.normalizer.update(state)
        state = torch.tensor(self.normalizer.normalize(state), dtype=torch.float32)

        for step in range(total_steps):

            # -------------------------------------------------
            # Select action
            # -------------------------------------------------
            if self.is_discrete:
                logits = self.agent.actor(state.unsqueeze(0))
                action_idx = torch.argmax(logits, dim=1).item()
                action_for_env = [float(action_idx)]     # ALWAYS wrap in list
            else:
                action = (
                    self.agent.select_action(state.unsqueeze(0))
                    .squeeze()
                    .detach()
                    .numpy()
                )

                # If the action is scalar, wrap into list
                if isinstance(action, float) or (hasattr(action, "ndim") and action.ndim == 0):
                    action_for_env = [float(action)]
                else:
                    action_for_env = action.tolist()

            # -------------------------------------------------
            # Step environment
            # -------------------------------------------------
            next_state, reward, terminated, truncated, _ = self.env.step(action_for_env)
            done = terminated or truncated

            # Reward scaling (small but smooth)
            reward = reward * 0.01

            # Normalize next_state
            self.normalizer.update(next_state)
            next_state_norm = self.normalizer.normalize(next_state)
            next_state_tensor = torch.tensor(next_state_norm, dtype=torch.float32)

            # -------------------------------------------------
            # Store transition (normalized)
            # -------------------------------------------------
            if self.is_discrete:
                self.real_buffer.add(
                    state.tolist(),
                    action_idx,
                    reward,
                    next_state_norm.tolist(),
                    done,
                )
            else:
                self.real_buffer.add(
                    state.tolist(),
                    action_for_env,
                    reward,
                    next_state_norm.tolist(),
                    done,
                )

            # -------------------------------------------------
            # Reset or continue
            # -------------------------------------------------
            if done:
                new_state, _ = self.env.reset()
                self.normalizer.update(new_state)
                state = torch.tensor(self.normalizer.normalize(new_state), dtype=torch.float32)
            else:
                state = next_state_tensor

            # -------------------------------------------------
            # Train dynamics model
            # -------------------------------------------------
            dyn_loss = self.train_dynamics_model()

            # -------------------------------------------------
            # Generate synthetic rollouts
            # -------------------------------------------------
            self.generate_model_rollouts(model_rollouts_per_step)

            # -------------------------------------------------
            # SAC update
            # -------------------------------------------------
            sac_real = self.agent.update(self.real_buffer)
            sac_model = self.agent.update(self.model_buffer)

            if step % 20 == 0:
                print(
                    f"Step {step} | DynLoss={dyn_loss} | SAC_real={sac_real} | SAC_model={sac_model}"
                )

        print("Training finished.")
