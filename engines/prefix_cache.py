"""Exact fixed-token prefix reuse; model-specific prefill is caller-owned."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass


def shared_token_prefix(prompts: list[list[int]], limit: int = 256) -> tuple[int, ...]:
    if not prompts:
        return ()
    prefix: list[int] = []
    for values in zip(*prompts):
        if len(prefix) >= limit or len(set(values)) != 1:
            break
        prefix.append(values[0])
    return tuple(prefix)


@dataclass
class PreparedPrefix:
    prompt: list[int]
    cache: object | None
    cached_tokens: int
    hit: bool


class FixedPrefixStore:
    """A bounded immutable-template cache, accessed under the model's lock.

    Every request gets an independent deep copy. Actual full prompt tokens are
    checked before slicing, so tokenization merges cannot change the request.
    Generated assistant tokens are never retained in the reusable template.
    """

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self._templates: OrderedDict[tuple, object] = OrderedDict()

    def prepare(self, prefix: tuple[int, ...], full_prompt: list[int], make_cache, prefill) -> PreparedPrefix:
        if not prefix or len(prefix) >= len(full_prompt) or tuple(full_prompt[: len(prefix)]) != prefix:
            return PreparedPrefix(full_prompt, None, 0, False)
        hit = prefix in self._templates
        if not hit:
            cache = make_cache()
            prefill(list(prefix), cache)
            # Failed prefill is never installed as a reusable entry.
            self._templates[prefix] = cache
            while len(self._templates) > self.capacity:
                self._templates.popitem(last=False)
        self._templates.move_to_end(prefix)
        return PreparedPrefix(
            list(full_prompt[len(prefix) :]), copy.deepcopy(self._templates[prefix]), len(prefix), hit
        )
