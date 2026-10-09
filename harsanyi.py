"""A tiny, model-agnostic implementation of Harsanyi interaction.

This file intentionally uses only the Python standard library.  The central
idea is that a model is summarized by a *value function* ``v(S)``: give the
function a coalition of players and it returns one number describing how well
that coalition performs.

For a coalition S, its Harsanyi interaction is

    I(S) = sum_{T subseteq S} (-1) ** (|S| - |T|) * v(T).

Mathematically this is the Möbius transform on the subset lattice.  The direct
formula is easy to read but repeatedly visits the same subsets.  The fast
transform below computes every interaction in O(n * 2**n) time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Callable, Sequence


# A coalition is represented by one bool per player.  For example, with three
# players, (True, False, True) means that players 0 and 2 are present.
Coalition = tuple[bool, ...]
ValueFunction = Callable[[Coalition], float]


@dataclass(frozen=True)
class HarsanyiResult:
    """All intermediate and final quantities of one attribution run.

    The three tuples share the same ordering.  Entry ``k`` corresponds to the
    coalition encoded by integer ``k``.  Player ``i`` is controlled by bit
    ``i`` of that integer, so for two players the order is:

        empty set, {player 0}, {player 1}, {player 0, player 1}.

    Keeping masks, raw values, and interactions together makes it difficult to
    accidentally compare entries that refer to different coalitions.
    """

    masks: tuple[Coalition, ...]
    values: tuple[float, ...]
    interactions: tuple[float, ...]


def all_coalitions(n_players: int) -> tuple[Coalition, ...]:
    """Enumerate every subset of ``n_players`` players in binary order.

    There are ``2**n_players`` subsets.  The integer subset index is useful
    because adding player ``i`` is exactly the operation of setting bit ``i``.
    """

    if n_players < 0:
        raise ValueError("n_players must be non-negative")

    return tuple(
        tuple(bool(subset_index & (1 << player)) for player in range(n_players))
        for subset_index in range(1 << n_players)
    )


def mobius_transform(values: Sequence[float]) -> tuple[float, ...]:
    """Convert all coalition values ``v(S)`` to Harsanyi interactions ``I(S)``.

    ``values`` must contain one entry for every coalition and therefore its
    length must be a power of two.  The expected order is the binary order used
    by :func:`all_coalitions`.

    Why the update works
    --------------------
    For each player, every coalition containing that player subtracts the
    otherwise identical coalition without that player.  After processing all
    players, inclusion-exclusion has been applied along every dimension of the
    subset lattice, which is precisely the Harsanyi/Möbius transform.

    Unlike a dense ``2^n x 2^n`` matrix, this algorithm does not require
    ``O(4^n)`` memory.
    """

    n_values = len(values)
    if n_values == 0 or n_values & (n_values - 1):
        raise ValueError("values must have a positive power-of-two length")

    n_players = int(log2(n_values))
    interactions = [float(value) for value in values]

    for player in range(n_players):
        player_bit = 1 << player

        for coalition_index in range(n_values):
            # Only coalitions containing this player need an update.
            if coalition_index & player_bit:
                without_player = coalition_index ^ player_bit
                interactions[coalition_index] -= interactions[without_player]

    return tuple(interactions)


def inverse_mobius_transform(interactions: Sequence[float]) -> tuple[float, ...]:
    """Reconstruct ``v(S)`` from interactions; useful for tests and debugging.

    Harsanyi interactions satisfy

        v(S) = sum_{T subseteq S} I(T).

    This inverse transform performs those subset sums in O(n * 2**n) time.
    """

    n_values = len(interactions)
    if n_values == 0 or n_values & (n_values - 1):
        raise ValueError("interactions must have a positive power-of-two length")

    n_players = int(log2(n_values))
    values = [float(interaction) for interaction in interactions]

    for player in range(n_players):
        player_bit = 1 << player
        for coalition_index in range(n_values):
            if coalition_index & player_bit:
                without_player = coalition_index ^ player_bit
                values[coalition_index] += values[without_player]

    return tuple(values)


def calculate_harsanyi(
    value_function: ValueFunction,
    n_players: int,
) -> HarsanyiResult:
    """Evaluate every coalition and return its Harsanyi interaction.

    Parameters
    ----------
    value_function:
        A callable mapping a bool tuple to a scalar value.  It can wrap a
        lookup table, a game, or a machine-learning model.
    n_players:
        Number of atomic players.  Runtime grows exponentially because exact
        Harsanyi interaction requires all ``2**n_players`` coalitions.
    """

    masks = all_coalitions(n_players)
    values = tuple(float(value_function(mask)) for mask in masks)
    interactions = mobius_transform(values)
    return HarsanyiResult(masks, values, interactions)

