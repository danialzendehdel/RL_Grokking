import math
import numpy as np
import matplotlib.pyplot as plt

class EpsilonScheduler:
    def __init__(self, epsilon_max=1.0, epsilon_min=0.01, decay_rate=0.01):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate

    def get_epsilon(self, episode: int) -> float:
        # Calculate epsilon using exponential decay
        epsilon = (self.epsilon_min +
                   (self.epsilon_max - self.epsilon_min) * math.exp(-self.decay_rate * episode))
        return epsilon

    def decay_schedule(self, max_steps, log_start=-2, log_base=10):
        # Calculate number of steps to decay
        decay_steps = int(max_steps * self.decay_rate)
        rem_steps = max_steps - decay_steps

        # Create a logarithmic decay over decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        # Normalize values to be between 0 and 1
        values = (values - values.min()) / (values.max() - values.min())
        # Scale values to the range [min_value, init_value]
        values = (self.epsilon_max - self.epsilon_min) * values + self.epsilon_min

        # Pad the remaining steps with the minimum value
        values = np.pad(values, (0, rem_steps), 'edge')

        return values


# Define testing parameters
max_episodes = 200
epsilon_max = 1.0
epsilon_min = 0.01
decay_rate = 0.05  # Adjust to see different decay behaviors

# Initialize scheduler
scheduler = EpsilonScheduler(epsilon_max=epsilon_max, epsilon_min=epsilon_min, decay_rate=decay_rate)

# Generate exponential decay schedule
exp_schedule = [scheduler.get_epsilon(episode) for episode in range(max_episodes)]

# Generate logarithmic decay schedule
log_schedule = scheduler.decay_schedule(max_episodes)

# Plot both schedules for comparison
plt.figure(figsize=(10, 6))
plt.plot(exp_schedule, label="Exponential Decay", color="blue")
plt.plot(log_schedule, label="Logarithmic Decay", color="orange")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Comparison of Epsilon Decay Schedules")
plt.legend()
plt.grid(True)
plt.show()
