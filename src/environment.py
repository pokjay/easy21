from collections import namedtuple
from typing import Callable, Tuple

import numpy as np

STICK = 0
HIT = 1
ACTION_SPACE = [
    STICK,
    HIT,
]
DEALER_HIT_MAX = 17

# min in inclusive, max is exclusive!
Range = namedtuple('Range', ['min', 'max'])
PLAYER_RANGE_FOR_ACTION = Range(1, 22)
DEALER_RANGE_FOR_ACTION = Range(1, 11)


class Easy21:
    def __init__(self):
        self.action_space = [
            STICK,
            HIT,
        ]
        self.state_max_bound = (
            DEALER_RANGE_FOR_ACTION.max,
            PLAYER_RANGE_FOR_ACTION.max,
        )

        self.state_min_bound = (
            DEALER_RANGE_FOR_ACTION.min,
            PLAYER_RANGE_FOR_ACTION.min,
        )

    def reset(self):
        """
        Reset the environment to the initial state:
        "At the start of the game both the player and the dealer draw one black
        card (fully observed)"

        :return: Observation consisting of (dealer card, player card)
        """
        return tuple(np.random.randint(low=1, high=11, size=(2,)))

    def _draw(self) -> Tuple[int, str]:
        """
        Draw a card according to easy21 rules:
        - Each draw from the deck results in a value between 1 and 10 (uniformly distributed)
          with a colour of red (probability 1/3) or black (probability 2/3).
        - There are no aces or picture (face) cards in this game
        :return: Tuple of card value and card color
        """
        card = np.random.randint(1, 11)
        color = np.random.choice(['b', 'r'], p=[2/3, 1/3])

        return card, color

    def _draw_and_update(self, prev_sum):
        """
        Draw a card from the deck and add or subtract from the current sum of the given cards
        The values of the cards are added (black cards) or subtracted (red cards)

        :param prev_sum: Previous sum of cards
        :return: New sum of cards after drawing a card from the deck
        """
        value, color = self._draw()
        value = -1*value if color == 'r' else value
        return prev_sum + value

    def step(self, s: Tuple[int, int], a: int) -> Tuple[Tuple[int, int], int, bool]:
        """
        Takes as input a state s (dealer’s first card 1–10 and the player’s sum 1–21),
        and an action a (hit or stick), and returns a sample of the next state s0
        (which may be terminal if the game is finished) and reward r

        :param s: Input state s of format (dealer’s first card 1-10, player’s sum 1–21)
        :param a: Action to perform: 0-stick or 1-hit
        :return: The next state, the reward and True/False if terminal state
        """

        dealer_card, player_sum = s

        # Player sticks
        if STICK == a:
            s1, reward, is_terminal = self._stick(dealer_card, player_sum)
        # Player hits
        elif HIT == a:
            s1, reward, is_terminal = self._hit(dealer_card, player_sum)
        else:
            raise ValueError('Unknown action value:', a)

        # Clip state to boundaries
        s1 = tuple(np.clip(s1, a_min=self.state_min_bound, a_max=self.state_max_bound))

        return s1, reward, is_terminal

    def _stick(self, dealer_card: int, player_sum: int) -> Tuple[Tuple[int, int], int, bool]:
        """
        If the player sticks then the dealer starts taking turns. The dealer always
        sticks on any sum of 17 or greater, and hits otherwise.
        If the dealer goes bust, then the player wins; otherwise, the outcome – win (reward +1),
        lose (reward -1), or draw (reward 0) – is the player with the largest sum.

        :param dealer_card: The dealers' card, between 1-10
        :param player_sum: The sum of the players' cards, between 1-21
        :return: The next state, the reward and True/False if terminal state
        """

        # Dealer hits until reaching sum of 17 or greater
        dealer_sum = self._draw_and_update(dealer_card)
        while dealer_sum < DEALER_HIT_MAX:
            dealer_sum = self._draw_and_update(dealer_sum)

        # Dealer didn't go bust, winner has higher sum
        if 1 <= dealer_sum <= 21:
            if player_sum > dealer_sum:  # Player wins
                return (dealer_sum, player_sum), 1, True
            elif player_sum == dealer_sum:  # Tie
                return (dealer_sum, player_sum), 0, True
            else:  # Dealer wins!
                return (dealer_sum, player_sum), -1, True
        else:  # Dealer goes bust
            return (dealer_sum, player_sum), 1, True

    def _hit(self, dealer_card: int, player_sum: int) -> Tuple[Tuple[int, int], int, bool]:
        """
        If the player’s sum exceeds 21, or becomes less than 1, then she “goes bust” and loses the game (reward -1)

        :param dealer_card: The dealers' card, between 1-10
        :param player_sum: The sum of the players' cards, between 1-21
        :return: The next state, the reward and True/False if terminal state
        """
        new_player_sum = self._draw_and_update(player_sum)

        # Player still not bust yet
        if 1 <= new_player_sum <= 21:
            return (dealer_card, new_player_sum), 0, False
        else:
            return (dealer_card, new_player_sum), -1, True


if __name__ == "__main__":
    env = Easy21()
    s0 = env.reset()

    a = 1  # hit
    is_terminal = False
    reward = 0

    random_policy = lambda state: np.random.choice(env.action_space)

    while is_terminal is False:
        s1, reward, is_terminal = env.step(s0, a)
        s0, a = s1, random_policy(s1)

        print(s1, is_terminal, reward)

