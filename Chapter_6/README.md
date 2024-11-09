# Exploring Q-learning, SARSA, Monte Carlo, and Double Q-learning: Practical Insights Beyond the Books

In the journey to mastering reinforcement learning (RL), foundational algorithms like **Q-learning**, **SARSA**, **Monte Carlo Control**, and **Double Q-learning** are essential building blocks. While many textbooks introduce these algorithms and their mechanics, several practical aspects and nuanced questions often remain unexplored. This blog post dives into these methods with hands-on insights, explaining their differences, tackling common questions, and highlighting practical challenges such as maximization bias, policy consistency, and convergence speed. Let’s delve into these four algorithms, discuss why they behave differently, and compare their performance in a "slippery walk" environment.

---

## 1. SARSA and Q-learning: On-Policy vs. Off-Policy Learning

### Question: Why do we pre-select an action in SARSA before entering the loop, while in Q-learning, we don’t take an initial action?

The difference between **SARSA** and **Q-learning** stems from their distinct learning paradigms: **on-policy** and **off-policy**.

- **SARSA (State-Action-Reward-State-Action)** is an **on-policy** algorithm, meaning it learns the Q-values based on the actions taken by the current policy. This requires selecting an action for each state as we proceed, ensuring that the Q-values are updated based on the policy’s actual behavior. Therefore, SARSA selects the **first action before entering the loop** to follow the policy consistently throughout the episode:
  
  $$
  Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma Q(s', a') - Q(s, a) \right]
  $$

  Here, $a'$ (the action in the next state $s'$) is selected based on the current policy, making the algorithm on-policy.

- **Q-learning**, on the other hand, is an **off-policy** method, as it updates Q-values using the **maximum possible action-value** in the next state, independent of the policy used to generate the experience. This allows Q-learning to approximate the optimal Q-values while following an exploratory policy. Hence, **we do not pre-select an action** for Q-learning, as it only needs the best Q-value in the next state:

  $$
  Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
  $$

In summary, SARSA selects the first action to follow the current policy, while Q-learning skips this step, aiming to learn the optimal policy irrespective of the exploratory behavior.

---

## 2. Q-learning’s Overestimation and Double Q-learning’s Solution

### Question: Why does Q-learning overestimate the value function, and how does Double Q-learning address this issue?

One of the limitations of Q-learning is that it often **overestimates the value function**. This occurs because Q-learning’s update rule takes the **maximum Q-value** in the next state across all actions. By always choosing the maximum, Q-learning unintentionally amplifies any positive biases, a phenomenon known as **maximization bias**.

> "Q-learning often overestimates the value function. On every step, we take the maximum over the estimates of the action-value function of the next state. But what we need is the actual value of the maximum action-value function of the next state."

To address this, **Double Q-learning** uses two separate Q-tables, `Q1` and `Q2`, which alternate between selecting actions and evaluating them. Double Q-learning’s update rule is:

$$
Q_1(s, a) \leftarrow Q_1(s, a) + \alpha \left[ R + \gamma Q_2(s', \arg\max_{a'} Q_1(s', a')) - Q_1(s, a) \right]
$$

or alternatively,

$$
Q_2(s, a) \leftarrow Q_2(s, a) + \alpha \left[ R + \gamma Q_1(s', \arg\max_{a'} Q_2(s', a')) - Q_2(s, a) \right]
$$

Double Q-learning reduces maximization bias by:
- **Action Selection**: Using `np.argmax` on one Q-table (e.g., `Q1`) to select the action for the next state.
- **Action Evaluation**: Evaluating the chosen action’s value using the other Q-table (e.g., `Q2`). This separation helps prevent the accumulation of positive bias.

> "Doing this isn’t only an inaccurate way of estimating the maximum value but also a more significant problem, given that these bootstrapping estimates, which are used to form TD targets, are often biased. The use of a maximum of biased estimates as the estimate of the maximum value is a problem known as maximization bias."

This approach provides a more accurate Q-function estimate by reducing overestimation bias, making Double Q-learning more reliable in stochastic environments.

---

## 3. Monte Carlo vs. Temporal Difference Methods

### Question: How does Monte Carlo control differ from SARSA and Q-learning in terms of learning updates?

Unlike SARSA and Q-learning, which update Q-values after every step (using **bootstrapping**), **Monte Carlo Control** waits until the end of each episode to calculate the **total return** for each state-action pair and uses this to update Q-values:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ G - Q(s, a) \right]
$$

where $G$ is the cumulative reward from the end of the episode. Monte Carlo methods can be slower because they require full episodes to complete:

> "One of the disadvantages of Monte Carlo methods is that they’re offline methods in an episode-to-episode sense."

Thus, Monte Carlo often has a slower convergence rate compared to SARSA and Q-learning, which use step-wise updates for faster feedback.

---

## 4. Comparing Performance and Convergence: Experiment Results

After implementing and running all four algorithms (Monte Carlo, SARSA, Q-learning, and Double Q-learning) in a "slippery walk" environment, we observed their convergence behavior. Here’s a summary of our findings:

- **Q-learning** demonstrated the fastest convergence due to its off-policy nature, directly approximating the optimal policy.
- **SARSA** converged more slowly than Q-learning, as it is on-policy and must follow the exploratory policy, leading to more conservative updates.
- **Monte Carlo Control** required entire episodes for updates, which slowed down convergence compared to both SARSA and Q-learning.
- **Double Q-learning** showed the slowest convergence due to alternating updates between two Q-tables, but it mitigated maximization bias effectively.

This comparison highlights that each algorithm has distinct strengths and limitations: Q-learning excels in speed, SARSA in policy adherence, Monte Carlo in complete return estimates, and Double Q-learning in bias reduction.

---

## Conclusion

In summary:
- **SARSA** and **Q-learning** balance policy fidelity and optimality through on-policy and off-policy approaches, respectively.
- **Double Q-learning** addresses maximization bias in Q-learning by separating action selection and evaluation, though it converges more slowly.
- **Monte Carlo Control** provides full return estimates per episode, beneficial when step-wise updates are impractical.

Understanding these practical aspects helps navigate the complexities of real-world RL tasks, where algorithmic trade-offs like speed vs. accuracy, on-policy vs. off-policy, and bias vs. variance are critical. Each algorithm has its strengths, and the choice depends on the task's specific needs.

This exploration sheds light on the nuances of RL algorithms beyond textbook explanations, providing practical insights for choosing the right method in various applications.