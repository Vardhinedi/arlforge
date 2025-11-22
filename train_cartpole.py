import gymnasium as gym
from arlforge.adaptive_agent import AdaptiveAgent

env = gym.make("CartPole-v1")

agent = AdaptiveAgent(env)

agent.train(total_steps=200, model_rollouts_per_step=5)
