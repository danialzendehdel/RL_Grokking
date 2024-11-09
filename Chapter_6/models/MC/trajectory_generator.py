from Utility import EpsilonScheduler
from Environment import SlipperyWalkSeven
import numpy as np
from itertools import count

def trajectory_generator(select_action, Q, epsilon, env, max_steps):

    done, trajectory = False, []

    while not done:
        # Reset environment
        state, _ = env.reset()

        for t in count():

            action = select_action(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            experience = (state, action, reward, next_state, done)
            trajectory.append(experience)

            if done:
                break

            if t > max_steps - 1:
                trajectory = []
                break

            state = next_state

    return np.array(trajectory, dtype=object)






