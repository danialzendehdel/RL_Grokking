from select import select

import numpy as np
from Utility.EpsilonScheduler import EpsilonScheduler
from tqdm import tqdm

def DQN(env,
        gamma=1.0,
        init_alpha=0.5,
        min_alpha=0.01,
        alpha_decay_rate=0.5,
        init_epsilon=1.0,
        min_epsilon=0.01,
        epsilon_decay_rate=0.9,
        n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []

    Q1 = np.zeros((nS, nA), dtype=np.float64)
    Q2 = np.zeros((nS, nA), dtype=np.float64)
    Q1_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    Q2_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    decay_alpha = EpsilonScheduler(epsilon_max=init_alpha, epsilon_min=min_alpha, decay_rate=alpha_decay_rate)
    alpha = decay_alpha.decay_schedule(n_episodes)

    decay_epsilon = EpsilonScheduler(epsilon_max=init_epsilon, epsilon_min=min_epsilon, decay_rate=epsilon_decay_rate)
    epsilon = decay_epsilon.decay_schedule(n_episodes)


    for e in tqdm(range(n_episodes), leave=True):

        state, _ = env.reset()
        done = False

        while not done:
            action = select_action(state, (Q1 + Q2)/2, epsilon[e])
            next_state, reward, done, _ = env.step(action)

            if np.random.randint(2):

                argmax_Q1 = np.argmax(Q1[next_state])
                td_target = reward + gamma * Q2[next_state][argmax_Q1] * (not done)
                Q1[state][action] += alpha[e] * (td_target - Q1[state][action])

            else:

                argmax_Q2 = np.argmax(Q2[next_state])
                td_target = reward + gamma * Q1[next_state][argmax_Q2] * (not done)
                Q2[state][action] += alpha[e] * (td_target - Q2[state][action])

            state = next_state
        Q1_track[e] = Q1.copy()
        Q2_track[e] = Q2.copy()
        pi_track.append(np.argmax((Q1 + Q2)/2, axis=1))

    Q = (Q1 + Q2)/2
    V = np.max(Q, axis=1)

    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return pi, V, Q1_track, Q2_track, pi_track