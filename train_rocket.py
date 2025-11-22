from arlforge.envs.rocket_env import RocketEnv
from arlforge.adaptive_agent import AdaptiveAgent

env = RocketEnv()

agent = AdaptiveAgent(env)

agent.train(total_steps=300, model_rollouts_per_step=10)
