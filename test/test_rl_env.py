import unittest

from rl_env import BlackJackRLEnviroment


def card(rank: str, suit: str = "Hearts") -> tuple[str, str]:
    return rank, suit


class TestBlackJackRLEnviroment(unittest.TestCase):
    def setUp(self) -> None:
        self.env = BlackJackRLEnviroment()

    def tearDown(self) -> None:
        self.env.close()

    def test_reset_returns_valid_observation(self) -> None:
        observation, info = self.env.reset(seed=123)

        self.assertEqual(info, {})
        self.assertTrue(self.env.observation_space.contains(observation))

        player_score, dealer_upcard, usable_ace = observation
        self.assertGreaterEqual(player_score, 4)
        self.assertLessEqual(player_score, 21)
        self.assertGreaterEqual(dealer_upcard, 1)
        self.assertLessEqual(dealer_upcard, 10)
        self.assertIsInstance(usable_ace, bool)

    def test_reset_with_same_seed_is_deterministic(self) -> None:
        first_env = BlackJackRLEnviroment()
        second_env = BlackJackRLEnviroment()

        try:
            first_observation, _ = first_env.reset(seed=123)
            second_observation, _ = second_env.reset(seed=123)

            self.assertEqual(first_observation, second_observation)
            self.assertEqual(first_env.player_deck, second_env.player_deck)
            self.assertEqual(first_env.dealer_deck, second_env.dealer_deck)
        finally:
            first_env.close()
            second_env.close()

    def test_reset_deals_two_cards_to_each_player(self) -> None:
        self.env.reset(seed=123)

        self.assertEqual(len(self.env.player_deck), 2)
        self.assertEqual(len(self.env.dealer_deck), 2)
        self.assertEqual(len(self.env.deck), 48)

    def test_card_value(self) -> None:
        self.assertEqual(self.env.card_value("Ace"), 1)
        self.assertEqual(self.env.card_value("2"), 2)
        self.assertEqual(self.env.card_value("9"), 9)
        self.assertEqual(self.env.card_value("10"), 10)
        self.assertEqual(self.env.card_value("Jack"), 10)
        self.assertEqual(self.env.card_value("Queen"), 10)
        self.assertEqual(self.env.card_value("King"), 10)

    def test_hand_value_uses_ace_as_eleven_when_possible(self) -> None:
        self.assertEqual(
            self.env.hand_value([card("Ace"), card("6")]),
            17,
        )

    def test_hand_value_changes_ace_to_one_when_needed(self) -> None:
        self.assertEqual(
            self.env.hand_value([card("Ace"), card("6"), card("10")]),
            17,
        )

    def test_hand_value_supports_multiple_aces(self) -> None:
        self.assertEqual(
            self.env.hand_value([card("Ace"), card("Ace"), card("9")]),
            21,
        )

    def test_hand_value_returns_bust_score(self) -> None:
        self.assertEqual(
            self.env.hand_value([card("10"), card("9"), card("5")]),
            24,
        )

    def test_has_usable_ace(self) -> None:
        self.assertTrue(
            self.env.has_usable_ace([card("Ace"), card("6")])
        )
        self.assertFalse(
            self.env.has_usable_ace([card("Ace"), card("6"), card("10")])
        )
        self.assertFalse(
            self.env.has_usable_ace([card("10"), card("6")])
        )

    def test_dealer_upcard_is_encoded_as_one_for_ace(self) -> None:
        self.env.player_deck = [card("10"), card("6")]
        self.env.dealer_deck = [card("Ace"), card("10")]

        observation = self.env._get_obs()

        self.assertEqual(observation, (16, 1, False))

    def test_hit_returns_non_terminal_transition_when_player_does_not_bust(
        self,
    ) -> None:
        self.env.player_deck = [card("10"), card("6")]
        self.env.dealer_deck = [card("6"), card("10")]
        self.env.deck = [card("Ace")]

        observation, reward, terminated, truncated, info = self.env.step(1)

        self.assertEqual(observation, (17, 6, False))
        self.assertEqual(reward, 0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {})

    def test_hit_ends_episode_when_player_busts(self) -> None:
        self.env.player_deck = [card("10"), card("9")]
        self.env.dealer_deck = [card("6"), card("10")]
        self.env.deck = [card("5")]

        observation, reward, terminated, truncated, info = self.env.step(1)

        self.assertEqual(observation, (24, 6, False))
        self.assertEqual(reward, -1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {})

    def test_dealer_hits_until_reaching_seventeen(self) -> None:
        self.env.player_deck = [card("10"), card("8")]
        self.env.dealer_deck = [card("10"), card("2")]
        self.env.deck = [card("5")]

        _, reward, terminated, truncated, _ = self.env.step(0)

        self.assertEqual(self.env.dealer_score, 17)
        self.assertEqual(reward, 1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_player_wins_when_dealer_busts(self) -> None:
        self.env.player_deck = [card("10"), card("8")]
        self.env.dealer_deck = [card("10"), card("6")]
        self.env.deck = [card("10")]

        _, reward, terminated, truncated, _ = self.env.step(0)

        self.assertEqual(self.env.dealer_score, 26)
        self.assertEqual(reward, 1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_push_returns_zero_reward(self) -> None:
        self.env.player_deck = [card("10"), card("8")]
        self.env.dealer_deck = [card("10"), card("8")]
        self.env.deck = []

        _, reward, terminated, truncated, _ = self.env.step(0)

        self.assertEqual(self.env.dealer_score, 18)
        self.assertEqual(reward, 0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_player_loses_when_dealer_has_higher_score(self) -> None:
        self.env.player_deck = [card("10"), card("7")]
        self.env.dealer_deck = [card("10"), card("8")]
        self.env.deck = []

        _, reward, terminated, truncated, _ = self.env.step(0)

        self.assertEqual(reward, -1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
