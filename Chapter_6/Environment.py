import gymnasium as gym
from gymnasium import spaces
import random


class SlipperyWalkSeven(gym.Env):
    def __init__(self):
        super(SlipperyWalkSeven, self).__init__()

        # Define action and state spaces
        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Discrete(9)  # 0-7: walkable, 8: goal

        # Initialize environment parameters
        self.initial_state = 0
        self.goal = 8
        self.state = self.initial_state
        self.done = False

    def reset(self, seed: int | None = None, options: dict | None = None):
        # Reset state and done flag
        self.state = self.initial_state
        self.done = False
        return self.state, {}

    def step(self, action: int):
        # Determine intended movement direction
        intended_move = 1 if action == 1 else -1  # +1 for right, -1 for left

        # Define movement outcomes with probabilities
        outcomes = [intended_move, 0, -intended_move]
        probabilities = [0.5, 0.333, 0.167]

        # Select the actual movement based on probabilities
        move = random.choices(outcomes, probabilities)[0]
        self.state += move

        # Ensure the state stays within bounds [0, 8]
        self.state = max(0, min(self.state, 8))

        # Determine if the goal state has been reached
        self.done = self.state == self.goal

        # Calculate reward
        reward = self.get_reward(self.state)

        return self.state, reward, self.done, {}

    def get_reward(self, state: int) -> float:
        # Reward is given only for reaching the goal
        return 1.0 if state == self.goal else 0.0
