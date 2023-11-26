from typing import Callable

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from environment import DEALER_RANGE_FOR_ACTION, PLAYER_RANGE_FOR_ACTION


def episode(env, policy: Callable):
    s0 = env.reset()

    a = 1  # hit
    is_terminal = False
    reward = 0

    episode_states, rewards, actions = [], [], []

    while is_terminal is False:
        s1, reward, is_terminal = env.step(s0, a)

        episode_states.append(s0)
        rewards.append(reward)
        actions.append(a)

        s0, a = s1, policy(s1)

    return episode_states, rewards, actions


def plot_value_function(V: np.ndarray) -> None:
    """
    Plot a given value function on a 3D surface using matplotlib
    Uses the ranges defined in the environment file for the plot ranges

    :param V: 2D Numpy Array where the rows are the dealers values and the columns are the players values,
        So for the state (dealer, player) -> V[dealer, player] = value
    :return: Nothing
    """
    X, Y = np.mgrid[
        range(DEALER_RANGE_FOR_ACTION.min, DEALER_RANGE_FOR_ACTION.max),
        range(PLAYER_RANGE_FOR_ACTION.min, PLAYER_RANGE_FOR_ACTION.max)
    ]
    Z = V[
        DEALER_RANGE_FOR_ACTION.min: DEALER_RANGE_FOR_ACTION.max,
        PLAYER_RANGE_FOR_ACTION.min: PLAYER_RANGE_FOR_ACTION.max,
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    _ = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    plt.title("V")
    plt.ylabel('Player')
    plt.xlabel('Dealer')

    plt.show()
