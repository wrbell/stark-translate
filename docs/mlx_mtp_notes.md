# Gemma-4 MTP / `-assistant` drafter on MLX (mlx-optiq 0.4.34)

Notes for the Mac finals path. Installed package is **mlx-optiq 0.4.34** (import name `optiq`, MIT, Thin Signal). Spec runtime lives at `optiq/runtime/spec/` and is **separate** from the Qwen MTP stack at `optiq/runtime/mtp/`.

Gemma-4 does not ship an in-weight MTP head. Google publishes a 4-layer Q-only **`-assistant`** drafter (`model_type: gemma4_assistant`). mlx-lm 0.31.3 cannot load it (`Model type gemma4_assistant not supported`; upstream [ml-explore/mlx-lm#1276](https://github.com/ml-explore/mlx-lm/issues/1276) still open). The repo's `--mts` flag currently passes that id to `mlx_lm.load` / `generate(draft_model=)`, which never works. The working Metal path is `optiq.runtime.spec`, wrapped by `engines/mlx_spec.py`.

This wrapper does **not** vendor optiq's loop. `spec_generate(prompt: str)` is usable once prompt ids are converted without double-BOS (see §4). Integration into `MLXGemmaEngine` / `dry_run_ab.py` is a separate change.

Cached drafter used below: `mlx-community/gemma-4-e4b-it-assistant-bf16` (snapshot `844e008e06ef5562bdb89428d851d0634d119dcd`, `model.safetensors` **159,138,208 bytes**). Cached target: `mlx-community/gemma-4-e4b-it-OptiQ-4bit` (42 layers, 18 KV-shared, hidden 2560).

---

## 1. Public API (0.4.34)

Source of truth: `optiq/runtime/spec/__init__.py` `__all__`.

```python
from optiq.runtime.spec import GemmaAssistantDrafter, spec_generate, SpecConfig, SpecEvent
```

`SpecStats` is defined in `runtime.py` but **not** re-exported.

### 1.1 `SpecConfig` (`runtime.py:29-35`)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `gamma` | `int` | `1` | Draft tokens per outer step (`γ >= 1`; `<1` raises `ValueError`) |
| `max_tokens` | `int` | `256` | Emitted-token cap (includes the prefill-greedy first token) |
| `eos_token_id` | `int \| None` | `None` | Single stop id. If `None`, `tokenizer.eos_token_id` |
| `accept_temp` | `float` | `0.0` | Must be `0.0`. Anything else raises `NotImplementedError` ("only greedy verify") |

### 1.2 `SpecEvent` (`runtime.py:38-44`)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `kind` | `Literal["token", "done"]` | required | Token vs terminal stats event |
| `token_id` | `int` | `-1` | Vocab id (`-1` on `done`) |
| `text` | `str` | `""` | Per-token `tokenizer.decode([tok_id])` on `token`; `str(stats)` on `done` |
| `from_draft` | `bool` | `False` | `True` iff this token was a drafted token that matched verify |

### 1.3 `SpecStats` (`runtime.py:47-62`, attached only as `done.text = str(stats)`)

| Field | Default | Meaning |
|---|---|---|
| `n_emitted` | `0` | Tokens yielded as `kind="token"` |
| `n_drafted` | `0` | Drafter forwards (`γ` per outer step, including rejected) |
| `n_accepted` | `0` | Draft tokens that matched target greedy |
| `n_target_calls` | `0` | Target forwards (1 prefill + 1 per outer step) |
| `elapsed_s` | `0.0` | Wall time of the whole `spec_generate` call |

Properties: `acceptance_rate = n_accepted / n_drafted` (0 if no drafts), `tokens_per_second = n_emitted / elapsed_s`.

### 1.4 `spec_generate(target, drafter, tokenizer, prompt: str, cfg=None) -> Iterator[SpecEvent]`

(`runtime.py:130-259`)

- `target`: mlx-lm `gemma4.Model` (has `.language_model.model`).
- `drafter`: `GemmaAssistantDrafter`.
- `tokenizer`: anything with `.encode`, `.decode`, `.eos_token_id`.
- `prompt`: **Python `str`**. Prefill is `mx.array([tokenizer.encode(prompt)])` — default HF `add_special_tokens=True`. Does **not** take token ids. Does **not** skip BOS when the string already starts with `bos_token` (mlx-lm `generate.py:691-694` does).
- Yields `kind="token"` events, then one `kind="done"`.

`optiq.serve.install_assistant_drafter(target_model_path, drafter_id)` patches `mlx_lm.server.stream_generate` onto this loop (γ hard-coded to 1 there). Its docstring still claims γ>1 raises; **the runtime implements γ>1** (`runtime.py:142-145, 188-246`).

### 1.5 `GemmaAssistantDrafter` (`drafters/gemma_assistant.py`)

**Loader**

```python
GemmaAssistantDrafter.from_pretrained(repo_or_path: str) -> GemmaAssistantDrafter
```

`huggingface_hub.snapshot_download` if `repo_or_path` is not a directory; reads `config.json`; `mx.load` of the first `*.safetensors`; `_rename_weights`; `load_weights(..., strict=False)`; `mx.eval(model.parameters())`.

**Forward** (keyword-only)

```python
logits, next_target_hidden = drafter.forward(
    last_token_emb=...,   # (1, 1, backbone_hidden)  target embed * embed_scale
    target_hidden=...,    # (1, 1, backbone_hidden)  post-final-norm hidden
    shared_kv=...,        # {"sliding_attention": (K, V), "full_attention": (K, V)}
    position=...,         # int, absolute RoPE position for Q
)
# logits: (vocab_size,)
# next_target_hidden: (1, 1, backbone_hidden)  post_projection output for γ>1 chaining
```

**Config dataclass** `GemmaAssistantConfig.from_dict(cfg)` reads `backbone_hidden_size` (top-level, 2560 on E4B), `text_config.*` (`hidden_size=256`, 4 layers, 4 heads, 2 KV heads, `head_dim=256`, `global_head_dim=512`, `layer_types`, `sliding_window=512`, RoPE thetas, `partial_rotary_factor`), plus centroid fields (`num_centroids=2048`, `centroid_intermediate_top_k=32` → `use_ordered_embeddings=True` on E4B).

**Internals not in `__all__`:** `extract_typed_kv`, `find_donor_layers` (`kv_view.py`); `_target_inner`, `_project_to_logits`, `_target_embed`, `_greedy`, `_trim_cache` (`runtime.py`).

---

## 2. How the drafter is attached to the target

### Hidden-state tap

Prefill / verify call the target's **inner** transformer, not `target(...)`:

```python
# runtime.py:70-78, 94-101
lm = target.language_model          # gemma4_text.Model
hidden = lm.model(ids, cache=cache) # Gemma4TextModel → RMSNorm(h), BEFORE lm_head
logits = lm.model.embed_tokens.as_linear(hidden)  # tied head; no logit softcap
```

That hidden is post-final-norm, same tensor mlx-lm would project. E4B PLE still runs inside `Gemma4TextModel.__call__` (`mlx_lm/models/gemma4_text.py:530-534`).

The first emitted token is greedy argmax of the prefill logits (`from_draft=False`). `last_target_hidden = hidden[:, -1:, :]` (last prompt position).

### Shared embeddings (not weight-tied)

The drafter has its **own** `model_embed_tokens` of shape `(262144, 256)` (used by the centroid head). Draft inputs do **not** use that table.

`_target_embed` (`runtime.py:104-117`) looks up the **target** `embed_tokens` and multiplies `embed_scale = sqrt(hidden_size)`:

```python
return emb(mx.array([[token_id]])) * scale   # (1, 1, 2560) on E4B
```

`forward` concatenates that scaled embedding with the target hidden and applies `pre_projection`:

```
(last_token_emb ∥ target_hidden)  →  pre_projection  →  256-d drafter space
```

Checkpoint tensor `pre_projection.weight` is **`[256, 5120]`** = `2 * backbone → draft_hidden`. `post_projection.weight` is `[2560, 256]`. After the 4 blocks + `model_norm`, `post_projection` maps back to backbone space for the next γ step.

`load_weights(..., strict=False)` is load-bearing: the module constructs `nn.Linear(backbone + draft_hidden, draft_hidden)` = **2816→256** (`gemma_assistant.py:307-315`) which does **not** match 5120. MLX `strict=False` skips shape checks and `Module.update`s the 5120-column checkpoint in (`mlx/nn/layers/base.py:135-137, 206`). Forward concat of two 2560-d tensors then matches. `strict=True` would raise.

### KV handling

The drafter is Q-only (`q_proj` / `q_norm` / `o_proj`; no `k_proj`/`v_proj`). K/V come from two **donor** layers of the target (`kv_view.py`):

1. Walk `model.layers` where `layer.self_attn.has_kv` (`layer_idx < num_hidden_layers - num_kv_shared_layers`). Cache list length = donor count, not 42.
2. Remember the last `sliding_attention` donor and last `full_attention` donor **in cache coordinates**.
3. Read those caches in **chronological** order (`_read_cache_temporal`).

E4B: 42 layers, `num_kv_shared_layers=18` → 24 donors. Shared tail layers reuse `previous_kvs` of the same type (`gemma4_text.py:433-442`), so "last donor of type T" is the K/V the last logical T layer would attend to.

- Full-attention donors: plain `KVCache`. Slice `keys[..., :offset, :]` to drop the 256-step allocation tail.
- Sliding donors: `RotatingKVCache`. Call `_temporal_order` then slice to `min(offset, max_size)`.

Sliding-window mask on the drafter (`gemma_assistant.py:245-257`): Q at `position` may attend to K in `[max(0, position - window + 1), position]`. Assumes the cache spans absolute positions `[position - k_len + 1, position]`.

### Cache trimming (γ>1 partial accept)

Verify feeds `[next_token] + drafts` (γ+1 new positions) into the live target cache. Longest matching prefix length `k_accept`. If `k_accept < γ`, `_trim_cache(caches, γ - k_accept)` (`runtime.py:245-246, 277-292`). Uses `cache.trim(n)` when `is_trimmable()`. `RotatingKVCache.is_trimmable()` is `offset < max_size` (`cache.py:542-543`) — **once the ring has wrapped, trim raises `RuntimeError`**. Church prompts (~33 tok) + 10–60 generated tokens stay under `sliding_window=512`, so wrap is not hit in production.

On full accept, the bonus token's K/V is **not** in cache (verify only added `next_token + γ drafts`). Next iteration feeds the bonus as `next_token`.

### Position handling

Each outer step:

```python
position = caches[0].offset - 1          # runtime.py:182
# draft k = 0..γ-1 uses position + k      # runtime.py:194
q = self.rope(q, offset=position)        # gemma_assistant.py:191
```

`caches[0]` is the first donor (sliding `RotatingKVCache` on E4B). After prefill of L tokens, `offset == L`, so Q is RoPE'd at **L−1** (last prompt token) while `last_token_emb` is the embedding of the just-emitted token, which lives at position **L** and is **not yet in cache**. Subsequent drafts increment `position` by 1 per chained step against a **frozen** KV snapshot (the new draft tokens are not written into target KV until verify).

Full-attention RoPE is mlx-lm `ProportionalRoPE` with `partial_rotary_factor=0.25` (rotate 128 of 512 dims, θ=1e6). Sliding uses default RoPE, θ=1e4, `head_dim=256`. K in the target cache is already rotated (gemma4_text applies RoPE before `cache.update_and_fetch`); the drafter rotates **Q only**.

---

## 3. How γ>1 works

`runtime.py:178-256`, header comment `runtime.py:1-11`.

1. Freeze `shared_kv = extract_typed_kv(target, caches)`.
2. For `k in range(γ)`: one `drafter.forward`. Draft token `d_k = argmax(draft_logits)`. Chain `draft_token_for_input = d_k` and `draft_hidden_for_input = post_projection(h)` (still in **target** hidden space so `pre_projection` stays valid).
3. One target forward on `[next_token, d_1, …, d_γ]`.
4. `gt[k] = argmax(logits[:, k:k+1, :])`. Accept while `d_k == gt[k]`. `gt[γ]` is the bonus / first rejection.
5. Emit accepted drafts (`from_draft=True`), then emit `gt[k_accept]` (`from_draft=False`). Stop on EOS or `max_tokens`.
6. Trim rejected suffix from the target cache.

Each outer step therefore emits `k_accept + 1` tokens (or `k_accept` if EOS hits inside the accepted prefix). Metal cost: the (γ+1)-token verify scales near-linearly with γ, so optiq's own blog defaults γ=1 (math prompt: γ=1 → 1.34×, γ=2 → 1.27×, γ=3 → 0.96×). Google's `generation_config.json` on the drafter sets `num_assistant_tokens: 6`.

---

## 4. EOS

Gemma-4 stop ids (drafter `generation_config.json` and tokenizer):

| Token | Id |
|---|---|
| `<eos>` | 1 |
| `<turn|>` (`<end_of_turn>`) | 106 |
| `<tool_response>` | 50 |

`spec_generate` compares emitted ids to **one** `eos_token_id` (`runtime.py:155-157, 173, 229, 248`). It never reads `tokenizer.eos_token_ids` / `_eos_token_ids`. If that single id is `1`, a natural `<turn|>` (106) does **not** stop the loop — generation runs to `max_tokens` (the same class of bug as the TranslateGemma EOS fix in `engines/mlx_engine.py`).

`engines/mlx_spec.spec_stream` stops on the full set even if optiq continues, and does not yield the EOS token (matches mlx-lm `stream_generate`).

---

## 5. Memory footprint of the drafter

On-disk E4B assistant bf16: **151.8 MiB** (159,138,208 bytes). Breakdown from the safetensors header (no GPU load):

| Tensor | Shape | ~MiB bf16 |
|---|---|---|
| `model.embed_tokens.weight` | `[262144, 256]` | 128.0 |
| 4 × (attn + MLP + norms) | 3× sliding 256-d, 1× full Q 512-d | ~20 |
| `pre_projection.weight` | `[256, 5120]` | 2.5 |
| `post_projection.weight` | `[2560, 256]` | 1.25 |
| `masked_embedding.centroids.weight` | `[2048, 256]` | 1.0 |
| `masked_embedding.token_ordering` | `[262144] I64` | 2.0 |

mlx-optiq's blog quotes **~700 MB RSS / ~3 s load** on M4 Pro — weights plus allocator, tokenizer (`tokenizer.json` 32 MiB), and Metal copies. Resident add-on next to OptiQ-4bit E4B is that RSS, not the 152 MiB file. Drafter never writes a KV cache (Q-only against target donors).

---

## 6. Threading / stream assumptions

- No lock inside `optiq.runtime.spec`. Single-threaded Python loop; every `mx.*` runs on the **calling** thread's MLX stream.
- mlx ≥ 0.31.2 uses thread-local streams. Lazy arrays are bound to the creating stream (`engines/mlx_engine.py` `materialize_mlx_model`). `from_pretrained` already `mx.eval`s drafter parameters; `load_gemma4_drafter` evals again if `.parameters()` exists. The **target** must be materialized on the same thread that will run `spec_stream`.
- Do not share one `spec_generate` / one target cache across threads. Verify mutates the target KV in place and trims it.
- `make_prompt_cache(target)` is created per call; no prompt-cache reuse with MTS (same constraint the engine already documents for `draft_model=`).
- Metal default stream. No CUDA graphs, no `mx.new_thread_unsafe_stream`. Compatible with the live pipeline's `ThreadPoolExecutor(max_workers=2)` only if STT and this translation worker own **separate** models/streams — not if they share the Gemma target.

---

## 7. Ranked suspects for ~31 % Metal acceptance vs 70–87 % llama.cpp

mlx-optiq blog ([Gemma-4 spec decoding](https://mlx-optiq.com/blog/gemma-spec-decoding), [MTP guide](https://mlx-optiq.com/docs/mtp)): **1.18× geomean, 31.4 % accept** at γ=1 on M4 Pro against 4-bit OptiQ. llama.cpp MTP on the same E4B assistant reports 70–87 %. Output is greedy; church prompts are ~33 tok in, 10–60 tok out. Ranked by how likely they are to actually drop accept, with the lines to patch.

### 1. RoPE position off-by-one for drafted Q — **highest**

- `optiq/runtime/spec/runtime.py:182` `position = caches[0].offset - 1`
- `optiq/runtime/spec/runtime.py:194` `position + k` for chained drafts
- `optiq/runtime/spec/drafters/gemma_assistant.py:189-191` `q = self.rope(q, offset=position)`
- Sliding mask assumes that Q sits on the last cached index: `gemma_assistant.py:247-250` `first_cached = position - k_len + 1`

After prefill of L tokens the just-emitted token is at absolute position L and is **not in KV**. Q is rotated at L−1. llama.cpp / Ollama's `assistant.go` (MIT, [ollama#15980](https://github.com/ollama/ollama/pull/15980)) is the reference optiq cites — compare their Q offset to `offset` vs `offset-1`. A one-position RoPE error against already-rotated K is enough to tank accept from ~80 % to ~30 % while still beating chance (vocab 262k).

**Patch experiment:** `position = caches[0].offset` (and `first_cached = position - k_len`) on a canary prompt; watch `SpecStats.acceptance_rate`.

### 2. Batched verify ≠ sequential greedy (bf16 / quant attention drift)

- `runtime.py:94-101` `_target_step` (one forward for γ+1 ids)
- `runtime.py:206-218` `verify_ids = [[next_token] + drafts]`
- Blog probe: `max|logit_AB[1] − logit_B_seq| = 0.68` (~1.3 % relative at logit ~51). Argmax flips → branch from `mlx_lm.generate` (token-at-a-time).

This is an mlx-lm Gemma-4 artifact, not unique to the drafter, but it **caps** accept and breaks byte-identity. Sequential verify (γ+1 target calls of 1 token) is the A/B to run in `tools/mts_acceptance_probe.py` if batched identity fails.

### 3. 4-bit OptiQ hidden / embed vs bf16-trained drafter

- `runtime.py:104-117` `_target_embed` (quantized `embed_tokens` × `embed_scale`)
- `runtime.py:70-78` `_target_inner` (4-bit backbone `norm(h)`)
- Drafter weights are bf16; `pre_projection` was trained on bf16 concat(emb, hidden).

Blog explicitly blames this plus (2) for the gap vs Ollama's fp16 target. **Patch experiment:** run the same loop against `mlx-community/gemma-4-e4b-it-bf16` (or 8-bit) and compare accept. If accept jumps toward llama.cpp, this is confirmed.

### 4. Sliding-window K/V order after `RotatingKVCache` concat-update + trim

- `optiq/runtime/spec/kv_view.py:70-92` `_read_cache_temporal`
- `mlx_lm/models/cache.py:431-447` `_temporal_order`
- `cache.py:449-467` `_update_concat` (used when verify S>1)
- `cache.py:542-548` `trim`: only `offset -= n; _idx -= n`, tensors keep a stale tail
- `runtime.py:277-292` trim-or-raise

Church lengths do not wrap the 512-window, so `_temporal_order`'s wrap branch is cold. Still check: after a γ=2/3 **partial** accept, `extract_typed_kv` chronological length should equal `offset`. A stale slot in the sliding donor would misalign RoPE vs mask (`first_cached`).

### 5. Donor-layer index vs Google's training donors

- `kv_view.py:25-50` `find_donor_layers`
- `gemma4_text.py:181-183` `has_kv`, `433-442` `previous_kvs`

"Last sliding donor / last full donor" among the 24 E4B donors may not be the pair the assistant was trained on (some ports use a fixed layer index from the model card). Dump `last_sliding, last_full` and compare to Ollama's donor selection.

### 6. `pre_projection` in_features constructed as 2816, checkpoint is 5120

- `gemma_assistant.py:307-315` `nn.Linear(backbone_hidden_size + hidden_size, …)`
- Comment on 305-306 says `2*backbone=5120`
- Forward concat `gemma_assistant.py:359` is 5120
- `from_pretrained` `load_weights(..., strict=False)` `gemma_assistant.py:438`

After load the weight **is** 5120, so this is probably **not** the 31 % bug. It will bite anyone who switches to `strict=True` or re-inits the Linear. Patch: `nn.Linear(2 * cfg.backbone_hidden_size, cfg.hidden_size, bias=False)`.

### 7. Double BOS on `tokenizer.encode(prompt)` (identity + RoPE shift)

- `runtime.py:162` `input_ids = mx.array([tokenizer.encode(prompt)])`
- Contrast mlx-lm `generate.py:691-694`

If `prompt_ids` already start with `<bos>` (id 2) and we `decode` then `encode` with `add_special_tokens=True`, every position shifts by 1 and accept collapses. `engines/mlx_spec.py` strips a leading BOS from the decoded string **and** pins the first `encode()` to the original ids so `spec_stream(prompt_ids)` matches `mlx_lm.generate(prompt=prompt_ids)`.

### 8. Single EOS id (run-on after `<turn|>`)

- `runtime.py:155-157, 173, 229, 248`
- Drafter `generation_config.json` `eos_token_id: [1, 106, 50]`

Does not change accept of *earlier* tokens; it inflates `n_emitted` with pad and can pollute byte-identity. Wrapper stops on the full set.

### 9. Centroid-head numerics (E4B only)

- `gemma_assistant.py:384-422` `argpartition` top-32 of 2048 centroids, sparse logits filled at `-1e30`

Tie order vs llama.cpp could flip a draft. Secondary: E4B uses this head (`use_ordered_embeddings: true`); a 26B/31B assistant would use a dense `lm_head`.

### 10. Per-token `tokenizer.decode([id])` (display only)

- `runtime.py:267-274`

SentencePiece leading-space artifacts. Not an accept bug. Wrapper uses `tokenizer.detokenizer` (`reset` / `add_token` / `last_segment` / `finalize`).

### 11. Missing logit softcap on verify (not a greedy bug)

- `runtime.py:81-91` vs `gemma4_text.py:606-607` `final_logit_softcapping` (default 30)

`cap * tanh(x/cap)` is per-component monotonic → greedy argmax unchanged. Ignore for this workload (`temperature=0`).

### What we did **not** vendor

`spec_generate`'s `prompt: str` signature is awkward (ids in, string expected, single EOS), but it is wrap-able. `engines/mlx_spec.py` keeps the optiq loop, converts ids → string, pins `encode` to the original ids, stops on the full EOS set, and detokenizes with mlx-lm's streaming detokenizer. If (1)+(2) need a sequential-verify fork, vendor a minimal MIT loop then (Ollama `mtp.go` / optiq `runtime.py`, both MIT) — not done here.

---

## 8. mlx-optiq 0.4.34 → 0.5.6 (`runtime.spec` API)

Sources: [mlx-optiq.com/changelog](https://mlx-optiq.com/changelog) (public mirror of the private-monorepo `CHANGELOG.md`) and [PyPI mlx-optiq 0.5.6](https://pypi.org/project/mlx-optiq/) (released 2026-09-05). Installed 0.4.34 released 2026-08-30.

**No `runtime.spec` / `GemmaAssistantDrafter` / `spec_generate` / `SpecConfig` API changes are listed between 0.4.34 and 0.5.6.** Gemma `-assistant` speculation shipped earlier (blog 2026-05-22; changelog entries around speculative decoding for Qwen MTP + Gemma `--drafter`). 0.4.33–0.5.6 are prune-experts, MCP pins, OptiQ Cloud Boost, OptiQ Code harness (`auto_verify`, `replace_lines` removed, git read-only), and eval/serve bugfixes. None mention drafter RoPE, KV view, γ, or `SpecEvent`.

### Upgrade advisable?

**No, not for MTS.** Staying on **0.4.34** is the right call until an upstream spec-acceptance patch exists.

| Would break / shift | Detail |
|---|---|
| Spec API | No documented change — wrapper should still import. Silent private-repo edits would not show up until runtime. |
| Acceptance | 0.5.6 does not advertise a Metal-accept fix. Upgrading will not close the 31 % vs 70–87 % gap. |
| `transformers` pin | mlx-optiq still requires `transformers<5.13` (since 0.4.33). Repo extras ask `transformers>=5.5`. Compatible today; a 5.13 bump in this repo would break `pip install mlx-optiq`. |
| Surface area | 0.5.x is Cloud Boost + Code-agent. Unrelated, larger dependency/behavior surface on a church Mac. |
| This worktree | Task forbids installing packages / touching the GPU. Do not upgrade here. |

Revisit when a changelog line names `optiq.runtime.spec` (RoPE offset, batched-verify identity, or `pre_projection` in_features). Then re-read `spec/__init__.py` `__all__` and `SpecConfig` fields before widening the wrapper.

---

## 9. Wrapper (`engines/mlx_spec.py`)

| Symbol | Role |
|---|---|
| `load_gemma4_drafter(repo_id)` | Lazy `GemmaAssistantDrafter.from_pretrained`; `mx.eval(parameters())` if present; `ImportError` → `RuntimeError` with install hint |
| `spec_stream(model, tokenizer, drafter, prompt_ids, *, gamma=1, max_tokens=64)` | `Iterator[SpecToken]` via a small stream object with `.stats: SpecStats` after iteration |
| `SpecToken(token_id, from_draft, text)` | Detokenizer `last_segment`, not `event.text` |
| `SpecStats(n_tokens, n_from_draft, n_verify_steps, accept_rate, gamma)` | `n_verify_steps = n_tokens − n_from_draft − 1`; `accept_rate = n_from_draft / max(n_verify_steps * gamma, 1)` |

GPU soak (not run in this worktree): `python tools/mts_acceptance_probe.py` over 8 theological canaries + `TEST_SENTENCES`, γ ∈ {1,2,3}, vs greedy `mlx_lm.generate`.
