import numpy as np

from src.environment import Easy21
from src.helpers import episode


if __name__ == "__main__":
    env = Easy21()
    random_policy = lambda state: np.random.choice(env.action_space)
    all_rewards = [episode(env=env, policy=random_policy)[1][-1] for j in range(1000)]

    print(f'{np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}')
