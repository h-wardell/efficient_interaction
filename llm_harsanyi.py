"""Minimal PyTorch adapter for token-level Harsanyi interaction.

The mathematical implementation lives in ``harsanyi.py``.  This adapter only
defines what a coalition and its value mean for a causal language model:

1. Every prompt token (or token group) is a player.
2. Tokens outside a coalition are replaced by baseline tokens.
3. ``v(S)`` is the target sequence's mean log-probability.
4. A fast Möbius transform converts all ``v(S)`` values to interactions.

The model is expected to follow the common Hugging Face causal-LM convention:
``model(input_ids=...).logits`` has shape ``[batch, sequence, vocabulary]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class TokenInteractionResult:
    """Token-subset masks, subset rewards, and Harsanyi interactions.

    All tensors are returned on CPU so that the result does not unnecessarily
    occupy accelerator memory after attribution finishes.
    """

    masks: torch.Tensor
    values: torch.Tensor
    interactions: torch.Tensor


def _all_masks(n_players: int, device: torch.device) -> torch.Tensor:
    """Return a ``[2^n, n]`` bool tensor in binary subset order."""

    subset_ids = torch.arange(1 << n_players, device=device)
    player_bits = 1 << torch.arange(n_players, device=device)
    return (subset_ids[:, None] & player_bits[None, :]) != 0


def _mobius_transform(values: torch.Tensor) -> torch.Tensor:
    """PyTorch version of the O(n * 2^n) fast Möbius transform."""

    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional tensor")

    n_values = values.numel()
    if n_values == 0 or n_values & (n_values - 1):
        raise ValueError("values must have a positive power-of-two length")

    n_players = n_values.bit_length() - 1
    interactions = values.clone()
    coalition_ids = torch.arange(n_values, device=values.device)

    for player in range(n_players):
        player_bit = 1 << player
        contains_player = (coalition_ids & player_bit) != 0
        with_player = coalition_ids[contains_player]
        without_player = with_player ^ player_bit

        # ``without_player`` entries never overlap the assignment destination,
        # so this vectorized in-place update is safe.
        interactions[with_player] -= interactions[without_player]

    return interactions


def _normalize_player_groups(
    player_groups: Sequence[Sequence[int]] | None,
    prompt_length: int,
) -> tuple[tuple[int, ...], ...]:
    """Validate groups and use one token per player when groups are omitted."""

    if player_groups is None:
        return tuple((position,) for position in range(prompt_length))

    groups = tuple(tuple(int(position) for position in group) for group in player_groups)
    if not groups:
        raise ValueError("player_groups must contain at least one player")
    if any(not group for group in groups):
        raise ValueError("every player group must contain at least one token")

    flat_positions = [position for group in groups for position in group]
    if any(position < 0 or position >= prompt_length for position in flat_positions):
        raise ValueError("player_groups contains a token position outside the prompt")
    if len(flat_positions) != len(set(flat_positions)):
        raise ValueError("a token position may belong to at most one player group")

    return groups


def _extract_logits(model_output: object) -> torch.Tensor:
    """Support both Hugging Face ModelOutput and tuple-style model outputs."""

    if hasattr(model_output, "logits"):
        logits = model_output.logits
    elif isinstance(model_output, (tuple, list)) and model_output:
        logits = model_output[0]
    else:
        raise TypeError("model output must expose .logits or store logits at index 0")

    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("model logits must have shape [batch, sequence, vocabulary]")
    return logits


def attribute_tokens(
    model: object,
    input_ids: torch.Tensor,
    baseline_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    player_groups: Sequence[Sequence[int]] | None = None,
    batch_size: int = 16,
) -> TokenInteractionResult:
    """Calculate exact Harsanyi interactions among prompt tokens.

    Parameters
    ----------
    model:
        A causal language model.  It must already live on the same device as
        ``input_ids``.
    input_ids / baseline_ids:
        Integer tensors with shape ``[1, prompt_length]``.  For each coalition,
        unselected prompt positions come from ``baseline_ids`` and selected
        positions come from ``input_ids``.
    target_ids:
        Desired continuation with shape ``[target_length]`` or
        ``[1, target_length]``.
    player_groups:
        Optional groups of prompt positions.  Grouped positions are switched on
        and off together.  Positions not listed in any group always remain at
        baseline.
    batch_size:
        Number of coalitions evaluated in one model forward pass.

    Notes
    -----
    This function substitutes baseline tokens; it does not delete positions or
    alter attention masks.  The exact attribution is exponential in the number
    of players because it evaluates every coalition.
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, prompt_length]")
    if baseline_ids.shape != input_ids.shape:
        raise ValueError("baseline_ids must have the same shape as input_ids")
    if input_ids.dtype != torch.long or baseline_ids.dtype != torch.long:
        raise TypeError("input_ids and baseline_ids must use torch.long")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if target_ids.ndim == 1:
        target_ids = target_ids.unsqueeze(0)
    if target_ids.ndim != 2 or target_ids.shape[0] != 1:
        raise ValueError("target_ids must have shape [target_length] or [1, target_length]")
    if target_ids.dtype != torch.long or target_ids.shape[1] == 0:
        raise ValueError("target_ids must be a non-empty torch.long tensor")
    if target_ids.device != input_ids.device or baseline_ids.device != input_ids.device:
        raise ValueError("all input tensors must be on the same device")

    prompt_length = input_ids.shape[1]
    groups = _normalize_player_groups(player_groups, prompt_length)
    n_players = len(groups)
    masks = _all_masks(n_players, input_ids.device)

    # Map player-level masks to token-level masks once.  This makes the batching
    # loop below a simple torch.where operation.
    token_masks = torch.zeros(
        masks.shape[0], prompt_length, dtype=torch.bool, device=input_ids.device
    )
    for player, positions in enumerate(groups):
        token_masks[:, list(positions)] = masks[:, player, None]

    all_values: list[torch.Tensor] = []
    target_length = target_ids.shape[1]

    with torch.inference_mode():
        for start in range(0, masks.shape[0], batch_size):
            stop = min(start + batch_size, masks.shape[0])
            current_token_masks = token_masks[start:stop]
            current_batch_size = stop - start

            original = input_ids.expand(current_batch_size, -1)
            baseline = baseline_ids.expand(current_batch_size, -1)
            masked_prompts = torch.where(current_token_masks, original, baseline)

            targets = target_ids.expand(current_batch_size, -1)
            model_input = torch.cat((masked_prompts, targets), dim=1)
            logits = _extract_logits(model(input_ids=model_input))

            # In a causal LM, logits at position prompt_length - 1 predict the
            # first target token.  The next position predicts the second token,
            # and so on.
            target_logits = logits[
                :, prompt_length - 1 : prompt_length + target_length - 1, :
            ]
            if target_logits.shape[1] != target_length:
                raise ValueError("model returned too few sequence positions")

            log_probs = torch.log_softmax(target_logits.float(), dim=-1)
            selected_log_probs = torch.gather(
                log_probs, dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)

            # A mean rather than a sum keeps the reward scale comparable when
            # target strings have different token lengths.
            all_values.append(selected_log_probs.mean(dim=1).cpu())

    values = torch.cat(all_values)
    interactions = _mobius_transform(values)
    return TokenInteractionResult(
        masks=masks.cpu(),
        values=values,
        interactions=interactions,
    )

