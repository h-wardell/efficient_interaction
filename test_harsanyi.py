"""Small tests that double as executable examples for the mathematical core."""

import unittest

from harsanyi import (
    all_coalitions,
    calculate_harsanyi,
    inverse_mobius_transform,
    mobius_transform,
)


class HarsanyiTest(unittest.TestCase):
    def test_two_player_synergy_example(self):
        # Empty=0, A=2, B=3, and A+B=9.  The pair therefore contributes four
        # units beyond what A and B already contribute independently.
        table = {
            (False, False): 0.0,
            (True, False): 2.0,
            (False, True): 3.0,
            (True, True): 9.0,
        }

        result = calculate_harsanyi(table.__getitem__, n_players=2)

        self.assertEqual(result.masks, all_coalitions(2))
        self.assertEqual(result.values, (0.0, 2.0, 3.0, 9.0))
        self.assertEqual(result.interactions, (0.0, 2.0, 3.0, 4.0))

    def test_transform_is_invertible(self):
        values = (1.0, 2.0, -3.0, 7.0, 0.5, 4.0, 8.0, -2.0)
        interactions = mobius_transform(values)
        reconstructed = inverse_mobius_transform(interactions)

        for actual, expected in zip(reconstructed, values):
            self.assertAlmostEqual(actual, expected)

    def test_efficiency_property(self):
        # The sum of every interaction contained in the full coalition equals
        # v(N).  For the full coalition, "contained" means every interaction.
        values = (0.0, 1.0, 2.0, 5.0, -1.0, 3.0, 4.0, 10.0)
        interactions = mobius_transform(values)
        self.assertAlmostEqual(sum(interactions), values[-1])

    def test_rejects_non_power_of_two_values(self):
        with self.assertRaises(ValueError):
            mobius_transform((1.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()

