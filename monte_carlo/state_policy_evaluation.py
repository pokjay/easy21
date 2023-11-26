from typing import Callable

import numpy as np

from environment import DEALER_RANGE_FOR_ACTION, Easy21, PLAYER_RANGE_FOR_ACTION
from helpers import episode


def _init_tables():
    """
    Initalize the tables needed for Monte Carlo evaluation:
        - N which stores the count of visits per state
        - S which stores the total reward per state
    :return: N and S initialized to 0's
    """
    table_shape = (DEALER_RANGE_FOR_ACTION.max, PLAYER_RANGE_FOR_ACTION.max)
    N = np.zeros(shape=table_shape)
    S = np.zeros(shape=table_shape)

    return N, S


def first_visit_mc_policy_eval(env, policy: Callable, n_episodes: int) -> np.ndarray:
    """
    Run First Visit Monte Carlo Policy Evaluation.
    Given an environment and policy, evaluate the policy by running episodes
    and calculating the mean reward per state.
    First visit only takes into account the first visit to each state per episode.

    :param env: Environment to run in, should have step and reset methods
    :param policy: Policy function, which given a state returns the action to perform
    :param n_episodes: Number of episodes to run
    :return: Numpy array representing the value function of the policy
    """
    N, S = _init_tables()

    all_episodes = [episode(env=env, policy=policy) for j in range(n_episodes)]

    for states, rewards, actions in all_episodes:
        visited_states = []
        G = 0
        for i in range(len(states)-1, -1, -1):
            dealer, player = states[i]
            G += rewards[i]
            if (dealer, player) not in visited_states:
                N[dealer, player] += 1
                S[dealer, player] += G
            visited_states.append((dealer, player))

    with np.errstate(divide='ignore', invalid='ignore'):
        V = S / N
        V = np.nan_to_num(V)

    return V


def every_visit_mc_policy_eval(env, policy, n_episodes):
    """
    Run Every Visit Monte Carlo Policy Evaluation.
    Given an environment and policy, evaluate the policy by running episodes
    and calculating the mean reward per state.
    Every visit takes into account all visits to each state.

    :param env: Environment to run in, should have step and reset methods
    :param policy: Policy function, which given a state returns the action to perform
    :param n_episodes: Number of episodes to run
    :return: Numpy array representing the value function of the policy
    """
    N, S = _init_tables()

    all_episodes = [episode(env=env, policy=policy) for j in range(n_episodes)]

    for states, rewards, actions in all_episodes:
        G = 0
        for i in range(len(states)-1, -1, -1):
            dealer, player = states[i]
            G += rewards[i]
            N[dealer, player] += 1
            S[dealer, player] += G

    with np.errstate(divide='ignore', invalid='ignore'):
        V = S / N
        V = np.nan_to_num(V)

    return V


def incremental_mc_policy_eval(env, policy, n_episodes):
    """
    Run Incremental Monte Carlo Policy Evaluation.
    Given an environment and policy, evaluate the policy by running episodes
    and calculating the mean reward per state.
    Incremental MC uses the incremental mean equation to perform updates to the mean incrementally after each episode.

    :param env: Environment to run in, should have step and reset methods
    :param policy: Policy function, which given a state returns the action to perform
    :param n_episodes: Number of episodes to run
    :return: Numpy array representing the value function of the policy
    """
    N, V = _init_tables()

    # Run episodes
    for j in range(n_episodes):
        states, rewards, actions = episode(env=env, policy=policy)
        G = 0
        # Update value function for each state visited in this episode
        for i in range(len(states)-1, -1, -1):
            dealer, player = states[i]
            G += rewards[i]
            idx = dealer, player

            N[idx] += 1
            alpha = 1 / N[idx]
            V[idx] = V[idx] + alpha * (G - V[idx])

    return V


if __name__ == "__main__":
    env = Easy21()
    random_policy = lambda state: np.random.choice(env.action_space)
    V = incremental_mc_policy_eval(env=env, policy=random_policy, n_episodes=1_000)

    print(V)
