from functools import partial

import numpy as np

from environment import ACTION_SPACE, DEALER_RANGE_FOR_ACTION, Easy21, PLAYER_RANGE_FOR_ACTION
from helpers import episode

rng = np.random.default_rng()


def _init_tables(env):
    """
    Initalize the tables needed for Monte Carlo evaluation:
        - N which stores the count of visits per state
        - Q which stores the total reward per state and action
    :return: N and Q initialized to 0's
    """
    table_shape = (DEALER_RANGE_FOR_ACTION.max, PLAYER_RANGE_FOR_ACTION.max, len(env.action_space))
    N = np.zeros(shape=table_shape)
    Q = np.zeros(shape=table_shape)

    return N, Q


def epsilon_greedy_policy(state, N, Q, N_0=100):
    """
    Policy which allows for exploration and exploitation.
    Flip a coin with epsilon probability to choose to explore, or take the greedy choice
    If taking the greedy choice, we take the argmax[Q(s, a)] for all actions
    If the actions have equal values, we flip a fair coin

    :param state: State tuple of (dealer card, player card)
    :param N: Table of count of visits per state
    :param Q: Table of total reward per state and action
    :param N_0: Constant for epsilon decay
    :return: Action to perform
    """

    # Exploration with probability epsilon
    epsilon = N_0 / (N_0 + N[state].sum())
    if rng.choice([True, False], p=[epsilon, 1.0 - epsilon]):
        return rng.choice(ACTION_SPACE)

    # If not exploring, use greedy(Q)
    # Special case if both actions have same value, argmax takes first, we want random
    arr = Q[state]
    if np.all(arr == arr.max()):
        return rng.choice(ACTION_SPACE)

    return np.argmax(arr)


def monte_carlo_control(env, policy, n_episodes):
    """
    Run Monte Carlo Control on environment using the given policy.
    Will perform iterative steps of policy evaluation and policy improvement based on evaluation results.

    :param env: Environment to run in, should have step and reset methods
    :param policy: Policy function, which given a state returns the action to perform
    :param n_episodes: Number of episodes to run
    :return: Numpy array representing the optimal state-action value function Q*
    """
    N, Q = _init_tables(env)

    policy = partial(policy, N=N, Q=Q)

    for _ in range(n_episodes):

        # Run episode
        states, rewards, actions = episode(env=env, policy=policy)
        G = 0

        # Update state-action value function for each state-value pair visited in this episode
        for idx in range(len(states)-1, -1, -1):
            G += rewards[idx]
            s_a = (*states[idx], actions[idx])

            N[s_a] += 1
            alpha = 1.0 / N[s_a]
            Q[s_a] = Q[s_a] + alpha * (G - Q[s_a])

    return Q


if __name__ == "__main__":
    env = Easy21()
    V = monte_carlo_control(env=env, policy=epsilon_greedy_policy, n_episodes=1_000)

    print(V)
