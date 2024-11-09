from models.MC.trajectory_generator import trajectory_generator
from Utility.EpsilonScheduler import EpsilonScheduler
import numpy as np
from tqdm import tqdm

def MonteCarlo(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_rate=0.01,
               init_epsilon=1.0,
               min_epsilon=0.01,
               epsilon_decay_rate=0.01,
               n_episodes=3000,
               max_steps=200,
               first_visit=True):

    nS, nA = env.observation_space.n, env.action_space.n

    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)

    decay_alpha = EpsilonScheduler(epsilon_max=init_alpha, epsilon_min=min_alpha, decay_rate=alpha_decay_rate)
    alpha = decay_alpha.decay_schedule(n_episodes)

    decay_epsilon = EpsilonScheduler(epsilon_max=init_epsilon, epsilon_min=min_epsilon, decay_rate=epsilon_decay_rate)
    epsilon = decay_epsilon.decay_schedule(n_episodes)

    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # Epsilon-greedy action selection
    select_action = lambda Q, state, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    for e in tqdm(range(n_episodes), leave=False):

        # Generate trajectory with the current epsilon
        trajectory = trajectory_generator(select_action, Q, epsilon[e], env, max_steps)

        # Track first visits for each state-action pair
        visited_states = np.zeros((nS, nA), dtype=bool)

        # Update Q-values for first-visit pairs
        for t, (state, action, reward, _, _) in enumerate(trajectory):

            if first_visit and visited_states[state][action]:
                continue
            visited_states[state][action] = True

            # Calculate discounted return (G) from this point
            n_steps = len(trajectory) - t
            G = np.sum(discounts[:n_steps] * np.array(trajectory[t:, 2]))

            # Update Q-value for this state-action pair
            Q[state][action] += alpha[e] * (G - Q[state][action])

        # Track the policy and Q-values over episodes
        Q_track[e] = Q.copy()
        pi_track.append(np.argmax(Q, axis=1))

    # Final policy and value extraction
    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return pi, V, Q_track, pi_track
