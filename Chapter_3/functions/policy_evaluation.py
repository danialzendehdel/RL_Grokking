import numpy as np

def policy_evaluation(pi, p, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(p), dtype=np.float64)  # for all s in S
    steps = 0
    while True:

        V = np.zeros(len(p), dtype=np.float64)

        for s in range(len(p)):
            # print(p[s][pi(s)])
            for prob, next_state, reward, done in p[s][pi(s)]:

                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))

        if np.max(np.abs(prev_V - V)) < theta:
            break
        steps += 1
        prev_V = V.copy()
    return V