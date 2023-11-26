import numpy as np

from environment import DEALER_RANGE_FOR_ACTION, Easy21, PLAYER_RANGE_FOR_ACTION
from helpers import episode


def first_visit_mc_policy_eval(env, policy, n_episodes):

    N = np.zeros(shape=(DEALER_RANGE_FOR_ACTION.max, PLAYER_RANGE_FOR_ACTION.max), dtype=float)
    S = np.zeros(shape=(DEALER_RANGE_FOR_ACTION.max, PLAYER_RANGE_FOR_ACTION.max), dtype=float)

    all_episodes = [episode(env=env, policy=policy) for j in range(n_episodes)]

    for states, rewards, actions in all_episodes:
        visited_states = []
        G = 0
        for i in range(len(states)-1, -1, -1):
            dealer, player = states[i]
            G += rewards[i]
            if (
                    (dealer, player) not in visited_states
                    and DEALER_RANGE_FOR_ACTION.min <= player < PLAYER_RANGE_FOR_ACTION.max
                    and DEALER_RANGE_FOR_ACTION.min <= dealer < DEALER_RANGE_FOR_ACTION.max
            ):
                N[dealer, player] += 1
                S[dealer, player] += G
                visited_states.append((dealer, player))

    with np.errstate(divide='ignore', invalid='ignore'):
        V = S / N
        V = np.nan_to_num(V)

    return V


def every_visit_mc_policy_eval(env, policy, n_episodes):

    N = np.zeros(shape=(DEALER_RANGE_FOR_ACTION.max, PLAYER_RANGE_FOR_ACTION.max), dtype=float)
    S = np.zeros(shape=(DEALER_RANGE_FOR_ACTION.max, PLAYER_RANGE_FOR_ACTION.max), dtype=float)

    all_episodes = [episode(env=env, policy=policy) for j in range(n_episodes)]

    for states, rewards, actions in all_episodes:
        G = 0
        for i in range(len(states)-1, -1, -1):
            dealer, player = states[i]
            G += rewards[i]
            if (
                    DEALER_RANGE_FOR_ACTION.min <= player < PLAYER_RANGE_FOR_ACTION.max
                    and DEALER_RANGE_FOR_ACTION.min <= dealer < DEALER_RANGE_FOR_ACTION.max
            ):
                N[dealer, player] += 1
                S[dealer, player] += G

    with np.errstate(divide='ignore', invalid='ignore'):
        V = S / N
        V = np.nan_to_num(V)

    return V


def incremental_mc_policy_eval(env, policy, n_episodes):
    N = np.zeros(shape=(DEALER_RANGE_FOR_ACTION.max, PLAYER_RANGE_FOR_ACTION.max), dtype=float)
    V = np.zeros(shape=(DEALER_RANGE_FOR_ACTION.max, PLAYER_RANGE_FOR_ACTION.max), dtype=float)

    # Run episodes
    for j in range(n_episodes):
        states, rewards, actions = episode(env=env, policy=policy)
        G = 0
        # Update value function for each state visited in this episode
        for i in range(len(states)-1, -1, -1):
            dealer, player = states[i]
            G += rewards[i]
            if (
                    DEALER_RANGE_FOR_ACTION.min <= player < PLAYER_RANGE_FOR_ACTION.max
                    and DEALER_RANGE_FOR_ACTION.min <= dealer < DEALER_RANGE_FOR_ACTION.max
            ):
                N[dealer, player] += 1

                alpha = 1 / N[dealer, player]
                V[dealer, player] = (
                        V[dealer, player] + alpha * (G - V[dealer, player])
                )

    return V


if __name__ == "__main__":
    env = Easy21()
    random_policy = lambda state: np.random.choice(env.action_space)
    # V = first_visit_mc_policy_eval(env=env, policy=random_policy, n_episodes=1_000)
    V = incremental_mc_policy_eval(env=env, policy=random_policy, n_episodes=1_000)

    print(V)
