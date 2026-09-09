"""Gemma-4 MTP / ``-assistant`` drafter speculative decoding for the MLX path.

Thin wrapper around ``optiq.runtime.spec`` (mlx-optiq >= 0.4.34). Optiq is
imported lazily so CI without Apple Silicon still collects tests.

Does not vendor optiq's loop — ``spec_generate`` is wrap-able once prompt ids
are converted without double-BOS. See ``docs/mlx_mtp_notes.md``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

OPTIQ_INSTALL_HINT = (
    "Gemma-4 MTS requires mlx-optiq (import name 'optiq') with "
    "optiq.runtime.spec. Install with: pip install 'mlx-optiq>=0.4.34'"
)


class SpecToken(NamedTuple):
    """One emitted token from ``spec_stream`` (EOS ids are not yielded)."""

    token_id: int
    from_draft: bool
    text: str


@dataclass
class SpecStats:
    """Acceptance / volume stats for one ``spec_stream`` call.

    For gamma=1, ``n_verify_steps = n_tokens - n_from_draft - 1`` (the leading
    ``-1`` is the prefill-greedy token) and ``accept_rate`` is accepted drafts
    per verify step. For gamma>1 the same verify-step count holds (each step
    still emits one non-draft correction/bonus plus 0..gamma drafts) and
    ``accept_rate = n_from_draft / max(n_verify_steps * gamma, 1)``.
    """

    n_tokens: int
    n_from_draft: int
    n_verify_steps: int
    accept_rate: float
    gamma: int


def compute_spec_stats(*, n_tokens: int, n_from_draft: int, gamma: int) -> SpecStats:
    """Derive verify-step count and accept rate from yielded tokens."""
    n_verify_steps = max(n_tokens - n_from_draft - 1, 0)
    accept_rate = n_from_draft / max(n_verify_steps * gamma, 1)
    return SpecStats(
        n_tokens=n_tokens,
        n_from_draft=n_from_draft,
        n_verify_steps=n_verify_steps,
        accept_rate=accept_rate,
        gamma=gamma,
    )


def _import_spec() -> tuple[Any, Callable[..., Iterator[Any]], Any]:
    """Lazy-import optiq.runtime.spec. Raises RuntimeError if missing."""
    try:
        from optiq.runtime.spec import GemmaAssistantDrafter, SpecConfig, spec_generate
    except ImportError as exc:
        raise RuntimeError(OPTIQ_INSTALL_HINT) from exc
    return GemmaAssistantDrafter, spec_generate, SpecConfig


def load_gemma4_drafter(repo_id: str) -> Any:
    """Load a Gemma-4 ``-assistant`` drafter via optiq.

    Calls ``GemmaAssistantDrafter.from_pretrained``, then ``mx.eval`` on
    ``drafter.parameters()`` when that method exists (from_pretrained already
    evals; this second eval is the cross-thread materialize the rest of the
    MLX path uses).
    """
    drafter_cls, _, _ = _import_spec()
    drafter = drafter_cls.from_pretrained(repo_id)
    params_fn = getattr(drafter, "parameters", None)
    if callable(params_fn):
        mx = importlib.import_module("mlx.core")
        mx.eval(params_fn())
    return drafter


def _as_int_list(prompt_ids: Any) -> list[int]:
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
        prompt_ids = prompt_ids[0]
    return [int(x) for x in prompt_ids]


def _eos_ids(tokenizer: Any) -> set[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids is None:
        ids = getattr(tokenizer, "_eos_token_ids", None)
    if ids is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        return {int(eos)} if eos is not None else set()
    return {int(x) for x in ids if x is not None}


def _prompt_ids_to_optiq_string(tokenizer: Any, prompt_ids: list[int]) -> str:
    """Decode ``prompt_ids`` for ``spec_generate``, avoiding a double BOS.

    optiq does ``tokenizer.encode(prompt)`` with HF's default
    ``add_special_tokens=True``, which prepends BOS. mlx-lm ``generate``
    skips that when the string already starts with ``bos_token``. If
    ``prompt_ids`` already begin with BOS we drop it before decode so encode
    re-adds it once. ``_PrefillEncodeTokenizer`` additionally pins the first
    ``encode()`` to the original ids so decode/encode drift cannot shift RoPE.
    """
    ids = list(prompt_ids)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and ids and ids[0] == int(bos_id):
        ids = ids[1:]
    try:
        text = tokenizer.decode(ids, skip_special_tokens=False)
    except TypeError:
        text = tokenizer.decode(ids)
    bos_tok = getattr(tokenizer, "bos_token", None)
    if isinstance(bos_tok, str) and bos_tok and text.startswith(bos_tok):
        text = text[len(bos_tok) :]
    return text


class _PrefillEncodeTokenizer:
    """Forward to ``tokenizer``, but first ``encode()`` returns ``prompt_ids``.

    Stops optiq's ``tokenizer.encode(prompt)`` from double-prepending BOS or
    otherwise disagreeing with the caller's token ids.
    """

    def __init__(self, tokenizer: Any, prompt_ids: list[int]):
        self._tokenizer = tokenizer
        self._prompt_ids = list(prompt_ids)
        self._encode_pinned = True

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        if self._encode_pinned:
            self._encode_pinned = False
            return list(self._prompt_ids)
        return self._tokenizer.encode(text, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


class SpecTokenStream:
    """Iterator of ``SpecToken``. ``stats`` is filled in ``finally`` when the
    generator is exhausted, closed, or garbage-collected after a ``break``.
    """

    def __init__(self, gamma: int):
        self.stats: SpecStats | None = None
        self._gamma = gamma
        self._inner: Iterator[SpecToken] | None = None

    def __iter__(self) -> Iterator[SpecToken]:
        if self._inner is None:
            raise RuntimeError("SpecTokenStream was not bound to a generator")
        return self._inner

    def _run(self, tokens: Iterator[SpecToken]) -> Iterator[SpecToken]:
        n_tokens = 0
        n_from_draft = 0
        try:
            for tok in tokens:
                n_tokens += 1
                if tok.from_draft:
                    n_from_draft += 1
                yield tok
        finally:
            self.stats = compute_spec_stats(
                n_tokens=n_tokens,
                n_from_draft=n_from_draft,
                gamma=self._gamma,
            )


def spec_stream(
    model: Any,
    tokenizer: Any,
    drafter: Any,
    prompt_ids: list[int],
    *,
    gamma: int = 1,
    max_tokens: int = 64,
) -> SpecTokenStream:
    """Speculatively decode ``prompt_ids`` with the Gemma-4 assistant drafter.

    Yields ``SpecToken``s built from ``tokenizer.detokenizer``, stopping on any
    id in ``tokenizer.eos_token_ids`` (or ``_eos_token_ids``) even if optiq
    itself only honors a single ``eos_token_id``. After the iterator ends,
    ``stream.stats`` is a ``SpecStats``.
    """
    _, spec_generate, spec_config_cls = _import_spec()
    ids = _as_int_list(prompt_ids)
    eos_ids = _eos_ids(tokenizer)
    preferred = getattr(tokenizer, "eos_token_id", None)
    if preferred is not None and int(preferred) in eos_ids:
        primary_eos = int(preferred)
    elif 1 in eos_ids:
        primary_eos = 1
    elif eos_ids:
        primary_eos = min(eos_ids)
    else:
        primary_eos = None
    prompt = _prompt_ids_to_optiq_string(tokenizer, ids)
    wrapped = _PrefillEncodeTokenizer(tokenizer, ids)
    cfg = spec_config_cls(
        gamma=gamma,
        max_tokens=max_tokens,
        eos_token_id=primary_eos,
        accept_temp=0.0,
    )

    detok = tokenizer.detokenizer
    detok.reset()

    def _tokens() -> Iterator[SpecToken]:
        pending: SpecToken | None = None
        n_yielded = 0
        try:
            for event in spec_generate(model, drafter, wrapped, prompt, cfg):
                kind = getattr(event, "kind", None)
                if kind == "done":
                    break
                if kind != "token":
                    continue
                token_id = int(event.token_id)
                if token_id in eos_ids:
                    break
                detok.add_token(token_id)
                tok = SpecToken(
                    token_id=token_id,
                    from_draft=bool(event.from_draft),
                    text=detok.last_segment,
                )
                if pending is not None:
                    yield pending
                    n_yielded += 1
                    if n_yielded >= max_tokens:
                        pending = None
                        break
                pending = tok
        except GeneratorExit:
            finalize = getattr(detok, "finalize", None)
            if callable(finalize):
                finalize()
            raise
        finalize = getattr(detok, "finalize", None)
        if callable(finalize):
            finalize()
        leftover = detok.last_segment if pending is not None else ""
        if pending is not None and n_yielded < max_tokens:
            if leftover:
                pending = SpecToken(pending.token_id, pending.from_draft, pending.text + leftover)
            yield pending

    stream = SpecTokenStream(gamma=gamma)
    stream._inner = stream._run(_tokens())
    return stream
