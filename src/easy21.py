import os
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


def cmp(a, b):
    return float(a > b) - float(a < b)


# 1-10 = Number cards
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def draw_card(np_random):
    """
    Draw a card according to easy21 rules:
    - Each draw from the deck results in a value between 1 and 10 (uniformly distributed)
      with a colour of red (probability 1/3) or black (probability 2/3).
    - Red results in subtracting the card value, while black results in adding the card value
    - There are no aces or picture (face) cards in this game
    :return: Tuple of card value and card color
    """
    color_mult = np.random.choice([1, -1], p=[2 / 3, 1 / 3])
    return color_mult * int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


class Player:
    def __init__(self):
        self.max = 22
        self.sum = 0

    def get_clipped_hand(self):  # Return current hand total
        return min(self.max, max(0, self.sum))

    def is_bust(self):  # Is this hand a bust?
        return not 0 < self.sum < self.max

    def score(self):  # What is the score of this hand (0 if bust)
        return 0 if self.is_bust() else self.sum

    def add_card(self, card):
        self.sum += card


class Dealer:
    def __init__(self):
        self.max = 11
        self.sum = 0
        self.first_card = 0

    def get_clipped_hand(self):  # Return current hand total
        return min(self.max, max(0, self.sum))

    def is_bust(self):  # Is this hand a bust?
        return not 0 < self.sum < self.max

    def score(self):  # What is the score of this hand (0 if bust)
        return 0 if self.is_bust() else self.sum

    def add_card(self, card):
        self.sum += card

    def set_first_card(self, card):
        self.first_card = card
        self.add_card(card)


def sum_hand(hand):  # Return current hand total
    return min(22, max(0, sum(hand)))


def is_bust(hand):  # Is this hand a bust?
    return not 0 < hand < 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else hand


class Easy21Env(gym.Env):
    """
    Easy21 is a card game designed for the assignment in David Silvers' Reinforcement Learning course
    which can be found at: [https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf](https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf)

    This class is largely based on the Blackjack Gymnasium environment, with small modifications made to update
    to Easy21 rules

    ## Description
    Below is the description of the game from the assignment:
    "Easy21 is similar to the Blackjack example in Sutton and Barto 5.3 – please note, however, that the rules of
    the card game are different and non-standard.

    The game is played with an infinite deck of cards (i.e. cards are sampled with replacement)
    - Each draw from the deck results in a value between 1 and 10 (uniformly distributed)
    with a colour of red (probability 1/3) or black (probability2/3).
    - There are no aces or picture (face) cards in this game
    - At the start of the game both the player and the dealer draw one black card (fully observed)
    - Each turn the player may either stick or hit
    - If the player hits then she draws another card from the deck
    - If the player sticks she receives no further cards
    - The values of the player’s cards are added (black cards) or subtracted (red cards)
    - If the player’s sum exceeds 21, or becomes less than 1, then she “goes bust” and loses the game (reward -1)
    - If the player sticks then the dealer starts taking turns. The dealer always sticks on any sum of 17 or greater,
    and hits otherwise. If the dealer goes bust, then the player wins; otherwise, the outcome – win (reward +1),
    lose (reward -1), or draw (reward 0) – is the player with the largest sum."

    ## Action Space
    The action shape is `(1,)` in the range `{0, 1}` indicating
    whether to stick or hit.

    - 0: Stick
    - 1: Hit

    ## Observation Space
    The observation consists of a 2-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10),
    and whether the player holds a usable ace (0 or 1).

    The observation is returned as `(int(), int())`.

    ## Starting State
    The starting state is initialised in the following range.

    | Observation               | Min  | Max  |
    |---------------------------|------|------|
    | Player current sum        |  -9   |  31  |
    | Dealer showing card value |  1   |  10  |

    ## Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0

    ## Episode End
    The episode ends if the following happens:

    - Termination:
    1. The player hits and the sum of hand falls below 1 or exceeds 21.
    2. The player sticks.

    ## Information

    No additional information is returned.

    ## Arguments

    ## References
    <a id="blackjack_ref"></a>[1] David Silvers' Reinforcement Learning course assignment: [https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf](https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf)

    ## Version History
    * v0: Initial version release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(22), spaces.Discrete(11), spaces.Discrete(2))
        )

        self.render_mode = render_mode

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.add_card(draw_card(self.np_random))
            if self.player.is_bust():
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
        else:  # stick: play out the dealers hand, and score
            terminated = True
            while 0 < self.dealer.sum < 17:
                self.dealer.add_card(draw_card(self.np_random))
            if self.dealer.is_bust():
                reward = 1.0
            else:
                reward = cmp(self.player.score(), self.dealer.score())

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return self.player.sum, self.dealer.sum

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.player = Player()
        self.dealer = Dealer()
        self.dealer.add_card(abs(draw_card(self.np_random)))
        self.player.add_card(abs(draw_card(self.np_random)))

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        player_sum, dealer_sum = self._get_obs()
        screen_width, screen_height = 600, 500
        card_img_height = screen_height // 3
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 20

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            gym_root = gym.__path__[0]
            toy_text = os.path.join(gym_root, 'envs', 'toy_text')
            image = pygame.image.load(os.path.join(toy_text, path))
            return image

        def get_font(path, size):
            gym_root = gym.__path__[0]
            toy_text = os.path.join(gym_root, 'envs', 'toy_text')
            font = pygame.font.Font(os.path.join(toy_text, path), size)
            return font

        small_font = get_font(
            os.path.join("font", "Minecraft.ttf"), screen_height // 15
        )
        dealer_text = small_font.render(
            "Dealer: " + str(dealer_sum), True, white
        )
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, spacing))

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        if 1 < self.dealer.first_card < 10:
            dealer_card_num = self.dealer.first_card
        elif 1 == self.dealer.first_card:
            dealer_card_num = 'A'
        else:
            dealer_card_num = 'J'
        dealer_card_img = scale_card_img(
            get_image(
                os.path.join(
                    "img",
                    f"S{dealer_card_num}.png",
                )
            )
        )
        dealer_card_rect = self.screen.blit(
            dealer_card_img,
            (
                screen_width // 2 - card_img_width - spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
        self.screen.blit(
            hidden_card_img,
            (
                screen_width // 2 + spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        player_text = small_font.render("Player", True, white)
        player_text_rect = self.screen.blit(
            player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing)
        )

        large_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 6)
        player_sum_text = large_font.render(str(player_sum), True, white)
        player_sum_text_rect = self.screen.blit(
            player_sum_text,
            (
                screen_width // 2 - player_sum_text.get_width() // 2,
                player_text_rect.bottom + spacing,
            ),
        )

        if self.dealer.is_bust():
            dealer_busted_text = small_font.render("Dealer busted!", True, white)
            dealer_bust_text_rect = self.screen.blit(
                dealer_busted_text,
                (
                    screen_width // 2 - dealer_busted_text.get_width() // 2,
                    dealer_card_rect.bottom + spacing // 2,
                ),
            )

        if self.player.is_bust():
            player_busted_text = small_font.render("Player busted!", True, white)
            player_bust_text_rect = self.screen.blit(
                player_busted_text,
                (
                    screen_width // 2 - player_busted_text.get_width() // 2,
                    player_sum_text_rect.bottom + spacing // 2,
                ),
            )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()


if "__main__" == __name__:
    env = Easy21Env(render_mode="human")
    env.reset()
    print()


# Pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)