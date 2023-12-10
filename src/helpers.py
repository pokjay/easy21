from typing import Callable, List, Tuple

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from src.environment import DEALER_RANGE_FOR_ACTION, PLAYER_RANGE_FOR_ACTION


def episode(env, policy: Callable) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Run an episode in the given environment using the given policy

    :param env: Environment to run in, should have step and reset methods
    :param policy: Policy function, which given a state returns the action to perform
    :return: List of the states, list of the rewards and list of the actions
    """
    s0 = env.reset()

    episode_states, rewards, actions = [], [], []
    is_terminal = False

    while is_terminal is False:
        a = policy(s0)
        s1, reward, is_terminal = env.step(s0, a)

        episode_states.append(s0)
        rewards.append(reward)
        actions.append(a)

        s0 = s1

    return episode_states, rewards, actions


def plot_value_function(V: np.ndarray, state_action_value_function=True, X_max=DEALER_RANGE_FOR_ACTION.max, Y_max=PLAYER_RANGE_FOR_ACTION.max, figsize=None) -> None:
    """
    Plot a given value function on a 3D surface using matplotlib
    Uses the ranges defined in the environment file for the plot ranges

    :param V: 2D or 3D Numpy Array where the rows are the dealers values and the columns are the players values.
        For state value function it is (dealer, player) -> V[dealer, player] = value
        For state action value function it is (dealer, player) -> V[dealer, player, action] = value
    :param state_action_value_function: True if using a state action value function, the max action value for each state
        will be plotted
    :param X_max: Maximal value for X axis
    :param Y_max: Maximal value for Y axis
    :return: Nothing
    """
    X, Y = np.mgrid[
        range(DEALER_RANGE_FOR_ACTION.min, X_max),
        range(PLAYER_RANGE_FOR_ACTION.min, Y_max)
    ]
    
    # If we have a State Action Value Function, we need to get the maximum value for each state our of all actions
    # So we want to reduce V[dealer, player, action] to V[dealer, player]
    if state_action_value_function is True:
        V = np.max(V, axis=2)
    
    Z = V[
        DEALER_RANGE_FOR_ACTION.min: X_max,
        PLAYER_RANGE_FOR_ACTION.min: Y_max,
    ]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    _ = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    plt.title("V")
    plt.ylabel('Player')
    plt.xlabel('Dealer')

    plt.show()
