# import numpy as np
# import matplotlib.pyplot as plt
# from Environment import SlipperyWalkSeven
# from Utility import EpsilonScheduler
# from models.MC.MontoCarlo import MonteCarlo
# from models.SARSA.SARSA import SARSA
# from models.Q_learning.Q_learning import Q_learning
#
#
# def main():
#     # Initialize environment and parameters
#     env = SlipperyWalkSeven()
#     gamma = 0.99
#     init_alpha = 0.5
#     min_alpha = 0.01
#     alpha_decay_rate = 0.01
#     init_epsilon = 1.0
#     min_epsilon = 0.01
#     epsilon_decay_rate = 0.01
#     n_episodes = 3000
#     max_steps = 200
#     first_visit = True
#
#     # Run Monte Carlo Control
#     # pi, V, Q_track, pi_track = MonteCarlo(
#     #     env=env,
#     #     gamma=gamma,
#     #     init_alpha=init_alpha,
#     #     min_alpha=min_alpha,
#     #     alpha_decay_rate=alpha_decay_rate,
#     #     init_epsilon=init_epsilon,
#     #     min_epsilon=min_epsilon,
#     #     epsilon_decay_rate=epsilon_decay_rate,
#     #     n_episodes=n_episodes,
#     #     max_steps=max_steps,
#     #     first_visit=first_visit
#     # )
#
#     pi, V, Q_track, pi_track = Q_learning(
#         env=env,
#         gamma=gamma,
#         init_alpha=init_alpha,
#         min_alpha=min_alpha,
#         alpha_decay_rate=alpha_decay_rate,
#         init_epsilon=init_epsilon,
#         min_epsilon=min_epsilon,
#         epsilon_decay_rate=epsilon_decay_rate,
#         n_episodes=n_episodes
#     )
#
#     # Plot state values over episodes
#     plot_value_function(Q_track, n_episodes)
#
#     # Print final state-action values (Q-table)
#     print_q_table(Q_track[-1])
#
#     # Plot the policy evolution over episodes (optional)
#     plot_policy(pi_track, env.observation_space.n)
#
#
# def plot_value_function(Q_track, n_episodes):
#     """Plots the maximum Q-values (state values) over episodes."""
#     state_values = [np.max(Q_track[ep], axis=1) for ep in range(n_episodes)]
#     avg_state_values = np.mean(state_values, axis=1)
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(avg_state_values)
#     plt.xlabel("Episode")
#     plt.ylabel("Average State Value")
#     plt.title("State Values Over Episodes")
#     plt.grid()
#     plt.show()
#
#
# def print_q_table(Q):
#     """Prints the final Q-table (state-action values) as a formatted table."""
#     print("Final State-Action Values (Q-table):")
#     print("State | Action 0 | Action 1")
#     print("-" * 25)
#     for state in range(Q.shape[0]):
#         print(f"{state:5} | {Q[state, 0]:8.3f} | {Q[state, 1]:8.3f}")
#
#
# def plot_policy(pi_track, n_states):
#     """Plots the evolution of the policy over episodes."""
#     policy_evolution = np.array(pi_track).T  # Transpose to get states as rows
#
#     plt.figure(figsize=(10, 6))
#     for state in range(n_states):
#         plt.plot(policy_evolution[state], label=f"State {state}")
#     plt.xlabel("Episode")
#     plt.ylabel("Action (0 = Left, 1 = Right)")
#     plt.title("Policy Evolution Over Episodes")
#     plt.legend(loc="upper right")
#     plt.grid()
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()


import numpy as np
import matplotlib.pyplot as plt
from Environment import SlipperyWalkSeven
from Utility.EpsilonScheduler import EpsilonScheduler
from models.MC.MontoCarlo import MonteCarlo
from models.SARSA.SARSA import SARSA
from models.Q_learning.Q_learning import Q_learning
from models.Q_learning.double_Q_learning import DQN  # Make sure this path points to your DQL implementation file

def main():
    # Initialize environment and parameters
    env = SlipperyWalkSeven()
    gamma = 0.99
    init_alpha = 0.5
    min_alpha = 0.01
    alpha_decay_rate = 0.01
    init_epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay_rate = 0.01
    n_episodes = 3000
    max_steps = 200
    first_visit = True

    # Run Monte Carlo Control
    _, _, mc_Q_track, _ = MonteCarlo(
        env=env,
        gamma=gamma,
        init_alpha=init_alpha,
        min_alpha=min_alpha,
        alpha_decay_rate=alpha_decay_rate,
        init_epsilon=init_epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay_rate=epsilon_decay_rate,
        n_episodes=n_episodes,
        max_steps=max_steps,
        first_visit=first_visit
    )

    # Run SARSA
    _, _, sarsa_Q_track, _ = SARSA(
        env=env,
        gamma=gamma,
        init_alpha=init_alpha,
        min_alpha=min_alpha,
        alpha_decay_rate=alpha_decay_rate,
        init_epsilon=init_epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay_rate=epsilon_decay_rate,
        n_episodes=n_episodes
    )

    # Run Q-learning
    _, _, qlearning_Q_track, _ = Q_learning(
        env=env,
        gamma=gamma,
        init_alpha=init_alpha,
        min_alpha=min_alpha,
        alpha_decay_rate=alpha_decay_rate,
        init_epsilon=init_epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay_rate=epsilon_decay_rate,
        n_episodes=n_episodes
    )

    # Run Double Q-learning
    _, _, q1_track, q2_track, _ = DQN(
        env=env,
        gamma=gamma,
        init_alpha=init_alpha,
        min_alpha=min_alpha,
        alpha_decay_rate=alpha_decay_rate,
        init_epsilon=init_epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay_rate=epsilon_decay_rate,
        n_episodes=n_episodes
    )
    # Average Q1 and Q2 to get the overall Q_track for DQL
    dql_Q_track = (q1_track + q2_track) / 2

    # Plot state values over episodes for each algorithm
    plot_comparison(mc_Q_track, sarsa_Q_track, qlearning_Q_track, dql_Q_track, n_episodes)


def plot_comparison(mc_Q_track, sarsa_Q_track, qlearning_Q_track, dql_Q_track, n_episodes):
    """Plots the average state values for Monte Carlo, SARSA, Q-learning, and Double Q-learning over episodes."""
    # Calculate the average state values across episodes for each algorithm
    mc_values = [np.mean(np.max(mc_Q_track[ep], axis=1)) for ep in range(n_episodes)]
    sarsa_values = [np.mean(np.max(sarsa_Q_track[ep], axis=1)) for ep in range(n_episodes)]
    qlearning_values = [np.mean(np.max(qlearning_Q_track[ep], axis=1)) for ep in range(n_episodes)]
    dql_values = [np.mean(np.max(dql_Q_track[ep], axis=1)) for ep in range(n_episodes)]

    # Plot the values
    plt.figure(figsize=(12, 8))
    plt.plot(mc_values, label="Monte Carlo Control", linestyle='--')
    plt.plot(sarsa_values, label="SARSA", linestyle='-.')
    plt.plot(qlearning_values, label="Q-learning", linestyle='-')
    plt.plot(dql_values, label="Double Q-learning", linestyle=':')
    plt.xlabel("Episode")
    plt.ylabel("Average State Value")
    plt.title("Comparison of Average State Values Over Episodes")
    plt.legend()
    plt.grid()
    plt.show()


def print_q_table(Q):
    """Prints the final Q-table (state-action values) as a formatted table."""
    print("Final State-Action Values (Q-table):")
    print("State | Action 0 | Action 1")
    print("-" * 25)
    for state in range(Q.shape[0]):
        print(f"{state:5} | {Q[state, 0]:8.3f} | {Q[state, 1]:8.3f}")


if __name__ == "__main__":
    main()
