
import numpy as np
from Utility.EpsilonScheduler import EpsilonScheduler
from tqdm import tqdm


def SARSA_lambda(env,
                 gamma=0.9,
                 init_alpha=0.5,
                 min_alpha=0.01,
                 alpha_decay_rate=0.5,
                 init_epsilon=1.0,
                 min_epsilon=0.01,
                 epsilon_decay_rate=0.9,
                 lambda_ = 0.9,
                 replacing_traces = True,
                 n_episodes=3000):


    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []  # Track the Greedy policy over episodes

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    E = np.zeros((nS, nA), dtype=np.float64)  # Eligibility traces

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    decay_alpha = EpsilonScheduler(epsilon_max=init_alpha, epsilon_min=min_alpha, decay_rate=alpha_decay_rate)
    alpha = decay_alpha.decay_schedule(n_episodes)

    decay_epsilon = EpsilonScheduler(epsilon_max=init_epsilon, epsilon_min=min_epsilon, decay_rate=epsilon_decay_rate)
    epsilon = decay_epsilon.decay_schedule(n_episodes)

    for e in tqdm(range(n_episodes), leave=False):

        E.fill(0)  # Reset eligibility traces
        state, _ = env.reset()
        done = False
        action = select_action(state, Q, epsilon[e])

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilon[e])

            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]

            E[state][action] += 1
            if replacing_traces:
                E.clip(0, 1, out=E)

            Q += alpha[e] * td_error * E
            E *= gamma * lambda_

            state, action = next_state, next_action

        Q_track[e] = Q.copy()
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return pi, V, Q_track, pi_track







