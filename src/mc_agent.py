import numpy as np

from src.environment import Easy21


class MonteCarloAgent:
    """
    Monte Carlo Control on environment using epsilon greedy policy.

    :param env: Environment to run in, should have step and reset methods
    :param N_0: Constant for epsilon decay
    :cvar N: Table of count of visits per state
    :cvar Q: Table of total reward per state and action
    :cvar rng: Numpy random number generator
    """
    def __init__(self, env, N_0=100):
        self.env: Easy21 = env
        self.N: np.ndarray = np.zeros(shape=(*self.env.state_max_bound, len(self.env.action_space)))
        self.Q: np.ndarray = np.zeros(shape=(*self.env.state_max_bound, len(self.env.action_space)))
        self.N_0: int = N_0
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
        self.dealer_states = []
        for _ in range(n_episodes):

            # Run episode
            s0 = self.env.reset()
            episode_observations = []
            is_terminal = False
            while is_terminal is False:
                a = self.epsilon_greedy_policy(s0)
                s1, reward, is_terminal = self.env.step(s0, a)
                episode_observations.append((s0, reward, a))
                s0 = s1

            if any([(x[0][0] == 2 and x[0][1] == 3) for x in episode_observations]):
                pass

            self.dealer_states.append([x[0][0] for x in episode_observations])

            # Update state-action value function for each state-value pair visited in this episode
            G = 0
            for idx in range(len(episode_observations) - 1, -1, -1):
                state, reward, action = episode_observations[idx]
                G += reward
                s_a = (*state, action)

                self.N[s_a] += 1.0
                alpha = 1.0 / self.N[s_a]
                self.Q[s_a] = self.Q[s_a] + alpha * (G - self.Q[s_a])

        return self.Q


if __name__ == "__main__":
    env = Easy21()
    mc = MonteCarloAgent(env=env)
    mc.learn(n_episodes=100_000)

    print(mc.Q)
