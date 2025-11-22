import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RocketEnv(gym.Env):
    """
    Simple 1D vertical rocket environment.
    State = [altitude, velocity, fuel]
    Action = throttle (0 to 1)
    Goal: reach 1000 meters without crashing
    """

    def __init__(self):
        super().__init__()

        # State: altitude, velocity, fuel
        self.observation_space = spaces.Box(
            low=np.array([0.0, -500.0, 0.0]),
            high=np.array([5000.0, 500.0, 100.0]),
            dtype=np.float32
        )

        # Action: throttle in [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        # Physics constants
        self.gravity = -9.8
        self.max_thrust = 30.0   # m/s² upward acceleration
        self.fuel_burn_rate = 0.5  # units per step
        self.dt = 0.1  # time step

        # Reset env
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.alt = 0.0
        self.vel = 0.0
        self.fuel = 100.0

        obs = np.array([self.alt, self.vel, self.fuel], dtype=np.float32)
        return obs, {}

    def step(self, action):
        throttle = float(np.clip(action[0], 0.0, 1.0))

        # Burn fuel
        if self.fuel > 0:
            thrust_acc = self.max_thrust * throttle
            self.fuel -= self.fuel_burn_rate * throttle
            if self.fuel < 0:
                self.fuel = 0
        else:
            thrust_acc = 0.0

        # Physics update
        acc = thrust_acc + self.gravity
        self.vel += acc * self.dt
        self.alt += self.vel * self.dt

        # Termination conditions
        terminated = False
        truncated = False

        # Crash
        if self.alt <= 0 and self.vel < -1:
            terminated = True
            reward = -100.0

        # Fuel empty + falling
        elif self.fuel <= 0 and self.vel < 0:
            terminated = True
            reward = -50.0

        # Goal: reach 1000 meters
        elif self.alt >= 1000:
            terminated = True
            reward = 200.0

        else:
            # Reward for going higher
            reward = self.alt * 0.01

        obs = np.array([self.alt, self.vel, self.fuel], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

