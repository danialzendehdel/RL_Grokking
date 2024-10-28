

import gymnasium as gym
import numpy as np
import random

from fontTools.subset import prune_hints

from Utilities.print_value_function import print_state_value_function, print_state_action_values, print_policy
from functions.policy_evaluation import policy_evaluation
from functions.policy_improvement import policy_improvement
from functions.policy_iteration import policy_iteration

desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]
env = gym.make("FrozenLake-v1", desc=desc, map_name=None, is_slippery=True)





P = env.unwrapped.P
# print(P)
#
# init_state = env.reset()
# goal_state = 15
#
# LEFT, DOWN, RIGHT, UP = range(4)
# careful_pi = lambda s: {
#     0:LEFT, 1:UP, 2:UP, 3:UP,
#     4:LEFT, 5:LEFT, 6:UP, 7:LEFT,
#     8:UP, 9:DOWN, 10:LEFT, 11:LEFT,
#     12:LEFT, 13:RIGHT, 14:RIGHT, 15:LEFT
# }[s]
#
# V, steps = policy_evaluation(careful_pi, P, gamma=0.99)
# pi, Q = policy_improvement(V, P, gamma=0.99)
#
# print_state_value_function(V, 4)
# print("\n")
# print_state_action_values(Q, V, 4)
# print(f"steps: {steps}")



optimal_V, optimal_pi = policy_iteration(P)


print_state_value_function(optimal_V, 4)
print("\n")
print_policy(optimal_pi, P, n_cols=4)