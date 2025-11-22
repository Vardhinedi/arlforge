import torch
from arlforge.core.agents.sac import SACAgent
from arlforge.utils.replay_buffer import ReplayBuffer

state_dim = 3
action_dim = 1

agent = SACAgent(state_dim, action_dim)
buffer = ReplayBuffer(100)

# Add two fake transitions
buffer.add([1,2,3], [0.5], 1.0, [1.1,2.1,3.1], False)
buffer.add([4,5,6], [-0.3], 0.8, [4.2,5.2,6.2], False)

# Update SAC:
log = agent.update(buffer, batch_size=2)
print("Update Log:", log)

# Test action selection
state = torch.tensor([[1.0,2.0,3.0]])
action = agent.select_action(state)
print("Selected action:", action)
