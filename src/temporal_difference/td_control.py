from functools import partial

import numpy as np

from src.environment import ACTION_SPACE, DEALER_RANGE_FOR_ACTION, Easy21, PLAYER_RANGE_FOR_ACTION
from src.helpers import episode

rng = np.random.default_rng()


def _init_tables(env):
    """
    Initalize the tables needed for Monte Carlo evaluation:
        - N which stores the count of visits per state
        - Q which stores the total reward per state and action
    :return: N and Q initialized to 0's
    """
    table_shape = (*env.state_max_bound, len(env.action_space))
    N = np.zeros(shape=table_shape)
    Q = np.zeros(shape=table_shape)

    return N, Q


def epsilon_greedy_policy(env, state, N, Q, N_0=100):
    """
    Policy which allows for exploration and exploitation.
    Flip a coin with epsilon probability to choose to explore, or take the greedy choice
    If taking the greedy choice, we take the argmax[Q(s, a)] for all actions
    If the actions have equal values, we flip a fair coin

    :param env: Environment to run in, should have step and reset methods
    :param state: State tuple of (dealer card, player card)
    :param N: Table of count of visits per state
    :param Q: Table of total reward per state and action
    :param N_0: Constant for epsilon decay
    :return: Action to perform
    """

    # State is out of bounds
    # if (np.array(state) >= np.array(env.state_max_bound)).any() or (np.array(state) <= np.array(env.state_min_bound)).any():
    #     return env.action_space[0]

    # Exploration with probability epsilon
    epsilon = N_0 / (N_0 + N[state].sum())
    if rng.choice([True, False], p=[epsilon, 1.0 - epsilon]):
        return rng.choice(env.action_space)

    # If not exploring, use greedy(Q)
    # Special case if both actions have same value, argmax takes first, we want random
    arr = Q[state]
    if np.all(arr == arr.max()):
        return rng.choice(env.action_space)

    return np.argmax(arr)


def sarsa(env, policy, lamda, gamma, n_episodes):
    """
    Run Monte Carlo Control on environment using the given policy.
    Will perform iterative steps of policy evaluation and policy improvement based on evaluation results.

    :param env: Environment to run in, should have step and reset methods
    :param policy: Policy function, which given a state returns the action to perform
    :param n_episodes: Number of episodes to run
    :return: Numpy array representing the optimal state-action value function Q*
    """
    table_shape = (*env.state_max_bound, len(env.action_space))
    Q = np.zeros(shape=table_shape)
    N = np.zeros(shape=table_shape)

    # policy = partial(policy, N=E, Q=Q)

    for i in range(n_episodes):

        E = np.zeros(shape=table_shape)

        s0 = env.reset()
        a0 = policy(env, s0, N=N, Q=Q, N_0=100)
        while True:
            # Take action A, observe R, S'
            s1, reward, is_terminal = env.step(s0, a0)
            s0_a0 = (*s0, a0)

            # Choose A' from S' using policy derived from Q
            if is_terminal:
                a1 = 0
                Q_s1_a1 = 0.0
            else:
                a1 = policy(env, s1, N=N, Q=Q, N_0=100)
                s1_a1 = (*s1, a1)
                Q_s1_a1 = Q[s1_a1]

            # Compute delta
            delta = reward + gamma * Q_s1_a1 - Q[s0_a0]
            E[s0_a0] += 1
            N[s0_a0] += 1

            # For all states and actions update Q
            alpha = 1.0 / N[s0_a0]
            Q += (alpha * delta * E)
            E = gamma * lamda * E

            s0 = s1
            a0 = a1

            if is_terminal is True:
                break

    return Q


if __name__ == "__main__":
    env = Easy21()
    Q = sarsa(env=env, policy=epsilon_greedy_policy, lamda=0, gamma=1, n_episodes=10_000)

    print(Q)
