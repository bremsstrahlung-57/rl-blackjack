from typing import Any, SupportsFloat, TypeVar

import gymnasium as gym
from gymnasium.core import RenderFrame

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class BlackJackRLEnviroment(gym.Env):
    def __init__(self) -> None:
        self.deck = []
        self.player_score       = 0
        self.dealer_score       = 0
        self.usable_aces        = False
        self.player_deck        = []
        self.dealer_deck        = []
        self.observation_space  = gym.spaces.Tuple((
            gym.spaces.Discrete(32),  # player score
            gym.spaces.Discrete(11),  # dealer upcard
            gym.spaces.Discrete(2),   # usable ace
        ))
        self.action_space       = gym.spaces.Discrete(2)
        self._actions           = {0: "stand", 1: "hit"}

    def _get_obs(self):
        self.player_score = self.hand_value(self.player_deck)
        self.usable_aces = self.has_usable_ace(self.player_deck)

        dealer_up_card = self.card_value(self.dealer_deck[0][0])

        return (
            self.player_score,
            dealer_up_card,
            self.usable_aces,
        )

    def _compute_reward(self):
        self.player_score = self.hand_value(self.player_deck)
        self.dealer_score = self.hand_value(self.dealer_deck)

        if self.dealer_score > 21 or  self.player_score >  self.dealer_score:
            return 1
        elif self.player_score < self.dealer_score:
            return -1
        else:
            return 0

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[ObsType, dict[str, Any]]:
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

        self.deck = [
            (card, suit) for suit in card_categories for card in cards_list
        ]
        self.np_random.shuffle(self.deck)

        self.player_deck = [self.draw_card(), self.draw_card()]
        self.dealer_deck = [self.draw_card(), self.draw_card()]

        self.player_score = self.hand_value(self.player_deck)
        self.dealer_score = self.hand_value(self.dealer_deck)

        obervation = self._get_obs()

        return obervation, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if action == 1:
            self.player_deck.append(self.draw_card())
            self.player_score = self.hand_value(self.player_deck)
            if self.player_score > 21:
                return self._get_obs(), -1, True, False, {}
            else:
                return self._get_obs(), 0, False, False, {}
        else:
            while self.hand_value(self.dealer_deck) < 17:
                self.dealer_deck.append(self.draw_card())
            self.dealer_score = self.hand_value(self.dealer_deck)

            reward = self._compute_reward()
            return self._get_obs(), reward, True, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return super().render()

    def close(self) -> None:
        return super().close()

    def draw_card(self) -> tuple[str, str]:
        return self.deck.pop()

    def card_value(self, card: str) -> int:
        if card == "Ace":
            return 1
        if card in ["Jack", "Queen", "King"]:
            return 10

        return int(card)

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

    def has_usable_ace(self, hand) -> bool:
        score_no_ace = sum(
            10 if c in ["Jack","Queen","King"] else (1 if c == "Ace" else int(c))
            for c, _ in hand
        )
        has_ace = any(c == "Ace" for c, _ in hand)
        return has_ace and score_no_ace + 10 <= 21
