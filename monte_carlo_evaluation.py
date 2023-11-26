import numpy as np

from environment import DEALER_RANGE_FOR_ACTION, Easy21, PLAYER_RANGE_FOR_ACTION
from helpers import episode


def _init_tables(env, state_action_value_function):
    if state_action_value_function is False:
        state_value_func_size = (DEALER_RANGE_FOR_ACTION.max, PLAYER_RANGE_FOR_ACTION.max)
        N = np.zeros(shape=state_value_func_size, dtype=float)
        S = np.zeros(shape=state_value_func_size, dtype=float)
    else:
        state_action_value_func_size = (DEALER_RANGE_FOR_ACTION.max, PLAYER_RANGE_FOR_ACTION.max, len(env.action_space))
        N = np.zeros(shape=state_action_value_func_size, dtype=float)
        S = np.zeros(shape=state_action_value_func_size, dtype=float)

    return N, S


def first_visit_mc_policy_eval(env, policy, n_episodes, state_action_value_function=True):
    N, S = _init_tables(env, state_action_value_function)

    all_episodes = [episode(env=env, policy=policy) for j in range(n_episodes)]

    for states, rewards, actions in all_episodes:
        visited_states = []
        G = 0
        for i in range(len(states)-1, -1, -1):
            dealer, player = states[i]
            G += rewards[i]
            if (dealer, player) not in visited_states:
                if state_action_value_function is False:
                    N[dealer, player] += 1
                    S[dealer, player] += G
                else:
                    N[dealer, player, actions[i]] += 1
                    S[dealer, player, actions[i]] += G
                visited_states.append((dealer, player))

    with np.errstate(divide='ignore', invalid='ignore'):
        V = S / N
        V = np.nan_to_num(V)

    return V


def every_visit_mc_policy_eval(env, policy, n_episodes, state_action_value_function=True):
    N, S = _init_tables(env, state_action_value_function)

    all_episodes = [episode(env=env, policy=policy) for j in range(n_episodes)]

    for states, rewards, actions in all_episodes:
        G = 0
        for i in range(len(states)-1, -1, -1):
            dealer, player = states[i]
            G += rewards[i]
            if state_action_value_function is False:
                N[dealer, player] += 1
                S[dealer, player] += G
            else:
                N[dealer, player, actions[i]] += 1
                S[dealer, player, actions[i]] += G

    with np.errstate(divide='ignore', invalid='ignore'):
        V = S / N
        V = np.nan_to_num(V)

    return V


def incremental_mc_policy_eval(env, policy, n_episodes, state_action_value_function=True, N=None, V=None):
    if N is None:
        N, V = _init_tables(env, state_action_value_function)

    # Run episodes
    for j in range(n_episodes):
        states, rewards, actions = episode(env=env, policy=policy)
        G = 0
        # Update value function for each state visited in this episode
        for i in range(len(states)-1, -1, -1):
            dealer, player = states[i]
            G += rewards[i]
            if state_action_value_function is True:
                idx = dealer, player, actions[i]
            else:
                idx = dealer, player

            N[idx] += 1
            alpha = 1 / N[idx]
            V[idx] = V[idx] + alpha * (G - V[idx])

    return V


def monte_carlo_control(env, n_episodes=1_000, N_0=100):
    N, Q = _init_tables(env, state_action_value_function=True)

    # Policy is greedy(Q)
    def policy(state):
        dealer, player = state

        # Exploration with probability 1 - epsilon
        epsilon = N_0 / (N_0 + N[dealer, player].sum())
        if np.random.choice([True, False], p=[1 - epsilon, epsilon]) is True:
            return np.random.choice(env.action_space)

        # If not exploring, use greedy(Q)
        return np.argmax(Q, axis=2)[dealer, player]

    # Evaluate policy using Incremental Monte Carlo Policy Evaluation
    Q = incremental_mc_policy_eval(env=env, policy=policy, n_episodes=n_episodes, N=N, V=Q)

    return Q


if __name__ == "__main__":
    env = Easy21()
    random_policy = lambda state: np.random.choice(env.action_space)
    # V = first_visit_mc_policy_eval(env=env, policy=random_policy, n_episodes=1_000)
    V = incremental_mc_policy_eval(env=env, policy=random_policy, n_episodes=1_000)

    print(V)
