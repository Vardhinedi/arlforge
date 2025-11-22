from arlforge.envs.rocket_env import RocketEnv

env = RocketEnv()
obs, _ = env.reset()
print("Initial state:", obs)

for i in range(10):
    obs, reward, terminated, truncated, _ = env.step([1.0])  # full throttle
    print(f"Step {i}: obs={obs}, reward={reward}")
    if terminated or truncated:
        print("Episode ended.")
        break
