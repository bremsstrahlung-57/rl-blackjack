import random
from typing import Optional

import gymnasium as gym
import numpy as np


class BlackJackRLEnviroment(gym.Env):
    def __init__(self) -> None:
        self.deck = []
        self.player_score = 0
        self.dealer_score = 0
        self.player_deck = []
        self.dealer_deck = []
        self._actions = {0: "stand", 1: "hit"}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        card_categories = ["Hearts", "Diamonds", "Clubs", "Spades"]
        cards_list = [
            "Ace",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "Jack",
            "Queen",
            "King",
        ]

        self.deck = [(card, suit) for suit in card_categories for card in cards_list]
        random.shuffle(self.deck)

        self.player_deck = [self.draw_card(), self.draw_card()]
        self.dealer_deck = [self.draw_card(), self.draw_card()]

        self.player_score = self.hand_value(self.player_deck)
        self.dealer_score = self.hand_value(self.dealer_deck)

        return self._get_obs(), {}

    def _get_obs(self):
        return (
            self.player_score,
            self.hand_value([self.dealer_deck[0]]),
            self.has_usable_ace(self.player_deck)
        )

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return super().render()

    def close(self) -> None:
        return super().close()

    def draw_card(self) -> tuple[str, str]:
        return self.deck.pop()

    def hand_value(self, hand) -> int:
        score = 0
        aces = 0
        for card, _ in hand:
            if card in ["Jack", "Queen", "King"]:
                score += 10
            elif card == "Ace":
                score += 11
                aces += 1
            else:
                score += int(card)
        while score > 21 and aces:
            score -= 10
            aces -= 1

        return score

    def has_usable_ace(self, deck):
        aces = 0
        for card, _ in deck:
            if card == "Ace":
                aces += 1

        return aces
