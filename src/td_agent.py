import numpy as np

from src.environment import Easy21


class SARSAAgent:
    """
    Temporal-Difference Control on environment using SARSA and epsilon greedy policy.

    :param env: Environment to run in, should have step and reset methods
    :param _lambda: Decaying factor, which gives less importance to the returns by larger n's
    :param gamma: Discount factor for decay of rewards
    :param N_0: Constant for epsilon decay
    :cvar N: Table of count of visits per state
    :cvar Q: Table of total reward per state and action
    :cvar rng: Numpy random number generator
    """
    def __init__(self, env: Easy21, lambda_: float, gamma: float=1.0, N_0: int = 100):
        self.env: Easy21 = env
        self.lambda_: float = lambda_
        self.gamma: float = gamma
        self.N_0: int = N_0
        self.N: np.ndarray = np.zeros(shape=(*self.env.state_max_bound, len(self.env.action_space)))
        self.Q: np.ndarray = np.zeros(shape=(*self.env.state_max_bound, len(self.env.action_space)))

        self.rng: np.random._generator.Generator = np.random.default_rng()

    def epsilon_greedy_policy(self, state):
        """
        Policy which allows for exploration and exploitation.
        Flip a coin with epsilon probability to choose to explore, or take the greedy choice
        If taking the greedy choice, we take the argmax[Q(s, a)] for all actions
        If the actions have equal values, we flip a fair coin

        :param state: State tuple of (dealer card, player card)
        :return: Action to perform
        """

        # Exploration with probability epsilon
        epsilon = self.N_0 / (self.N_0 + self.N[state].sum())
        if self.rng.choice([True, False], p=[epsilon, 1.0 - epsilon]):
            return self.rng.choice(self.env.action_space)

        # If not exploring, use greedy(Q)
        # Special case if both actions have same value, argmax takes first, we want random
        arr = self.Q[state]
        if np.all(arr == arr.max()):
            return self.rng.choice(self.env.action_space)

        return np.argmax(arr)

    def learn(self, n_episodes):
        """
        Run Monte Carlo Control on environment using epsilon greedy policy.
        Will perform iterative steps of policy evaluation and policy improvement based on evaluation results.

        :param n_episodes: Number of episodes to run
        :return: Numpy array representing the optimal state-action value function Q*
        """
        for i in range(n_episodes):
            E = np.zeros(shape=(*self.env.state_max_bound, len(self.env.action_space)))

            # Begin current episode
            s0 = self.env.reset()
            is_terminal = False

            # Get action on initial state
            a0 = self.epsilon_greedy_policy(s0)

            while is_terminal is False:
                # Take action a0, observe r, s1
                s1, reward, is_terminal = self.env.step(s0, a0)
                s0_a0 = (*s0, a0)

                # Choose A' from S' using policy derived from Q
                if is_terminal:
                    a1 = 0
                    Q_s1_a1 = 0.0
                else:
                    a1 = self.epsilon_greedy_policy(s0)
                    s1_a1 = (*s1, a1)
                    Q_s1_a1 = self.Q[s1_a1]

                # Compute delta
                delta = reward + self.gamma * Q_s1_a1 - self.Q[s0_a0]
                E[s0_a0] += 1
                self.N[s0_a0] += 1

                # For all states and actions update Q
                alpha = 1.0 / self.N[s0_a0]
                self.Q += (alpha * delta * E)
                E = self.gamma * self.lambda_ * E

                s0 = s1
                a0 = a1


if __name__ == "__main__":
    env = Easy21()
    sarsa = SARSAAgent(env = env, lambda_=1.0)
    sarsa.learn(n_episodes=1_000)

    print(sarsa.Q)
