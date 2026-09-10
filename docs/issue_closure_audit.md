# Issue closure audit — #131–#138, #176, #177 (correction)

Independent evidence review. No GitHub mutations. Original acceptance is the
issue body. Later comments are not a license to weaken it.

**Corrections vs first audit (`f2b42c5`):** that write-up treated local HEAD as
unpushed, claimed `git branch -r --contains 5154fb9` empty, and described a
controlled English rehearsal as hymn + speech. Those remote and hymn claims
are wrong. This revision uses live `gh` topology and the original checkout
read-only.

## Topology (inspected 2026-09-10 via `gh`, not stale worktree remotes)

| Ref | SHA | Fact |
|---|---|---|
| `origin/main` | `09e4679a041fa9825d397436496d6f30cd49aa08` | v2026.13. PRs #180–#191 merged. **Not** this delivery. |
| `origin/codex/mac-reliability-roadmap` | `5154fb9d6a4cebae7918d063f48df39814c6bc12` | Pushed. Draft **[PR #192](https://github.com/wrbell/stark-translate/pull/192)** `main` ← this branch. `isDraft: true`, `mergeable: MERGEABLE`, **not merged**. |
| PR #192 checks | commitlint, lint, test 3.11, test 3.12, security **pass**; audit **skipped** | Baseline CI green on the draft head. |
| Original checkout `/Users/willem/Code/vibes/SRTranslate` | `53d8680f75fce18d5110cca482f15c9984bf74de` | 9 commits **ahead** of the pushed PR head (through `6935ecc`, `7ef4b4c`, `53d8680`). |
| This worktree | audit-only | Must not be treated as the integration branch. |

Publication: source push / issue updates / merge are authorized by the user.
**PyPI and release-tag publishing remain pending by user choice.**

### Integration SHAs that matter (original checkout, not yet on PR #192)

| SHA | What |
|---|---|
| `5154fb9` | PR head. Worker rewrite (`f58f54a`) is here. Live `--mts` still swallows `mlx_lm.load` failure. |
| `6935ecce3011ba7f07240f79accee1748bc1a285` | Explicit live `--mts` reject; `tests/test_worker_translation_contract.py`. |
| `7ef4b4c5cec6da2573420582ca99670d92bd9c17` | Code SHA recorded on the prefix smoke. |
| `53d8680` | Current original-checkout HEAD. |

## Close policy (original AC only)

| Issue | On `main` now? | On PR #192 (`5154fb9`)? | After integrating through `53d8680`? |
|---|---|---|---|
| **#176** multiprocess | No (TG prompts still on `main`) | Implementation yes; **no** Pipe tests | **Yes**, option (a) + Pipe tests |
| **#177** MTP | Wrapper + failed probe; live `--mts` still swallows | Same as `main` for the live flag | **Conditional yes** as experimented-off **after** explicit reject lands. Speed gate **did not pass**. |
| **#131** startup | Partial Mac recon comments only | File operator sessions + failed mic | **No** until live mic EN and ES (deferred by user to tomorrow) |
| **#134** rehearsal | No written note on `main` | Laptop file sessions | **No**: spoken clip yes; **full hymn not captured**; TTFC not in a written runbook note |
| **#132** TTS | Code+tests (#188) | Same | **No**: no second output / virtual cable |
| **#137** corrections | CLI only | Review UI + unapproved note | **No**: no approved pair; no Sunday merge/retrain |
| **#135** W16/v2-cpo | No | No | **No** (WSL / live A/B) |
| **#136** Jacobo CPO | No | Church prompt ≠ CPO | **No** |
| **#138** Hindi | `hi` names only | Offline text probe | **No** live `hi` / church audio |
| **#133** diarize | Code (#189); gate unrun | Same | **No** two-speaker + p95 |

Do not add church hardware or live-mic as extra #134 requirements. Do not
treat fixtures, prefix-cache smoke, or file replay as those gates.

---

## #176 — `--multiprocess` TranslateGemma prompts

https://github.com/wrbell/stark-translate/issues/176

**Original:** `workers.py` hardcodes TranslateGemma chat + prompt cache, so
`--multiprocess` would feed Gemma 4 a TG prompt. Fix (a) shared
`build_chat_messages` / `ensure_stop_tokens` and skip gemma4 cache, or (b)
deprecate the flag.

**`main`:** still the bug
([`workers.py` @ `09e4679`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/workers.py)).

**PR #192 (`5154fb9`):** option (a). `translation_worker_main` builds
`MLXGemmaEngine(..., model_family=...)`. Parent passes `MODEL_FAMILY`.

**Original checkout `6935ecc`:**
`tests/test_worker_translation_contract.py` drives a real
`multiprocessing.Pipe` loop (thread target, no weights) for
`gemma4`/`translategemma` × EN/ES. Gemma 4 prompts are plain instruct strings
with `enable_thinking=False`; TG still uses `source_lang_code`. A/B six-tuple
contract held. **This file is not in `5154fb9`.**

**Close when:** `6935ecc` (or equivalent) is on the branch that merges, then
close. Suggested comment:

> Closing #176: option (a). Workers use `MLXGemmaEngine` with the parent
> `model_family`. Pipe tests in `tests/test_worker_translation_contract.py`
> (`6935ecc`) check shared prompts. `--multiprocess` remains off by default.

---

## #177 — Gemma 4 MTP / `--mts`

https://github.com/wrbell/stark-translate/issues/177

**Original gate:** identical vs greedy; canary ≥ 7/8; medium p50 ≤ 0.85×
post-EOS E4B; acceptance ≥ 30%; else `mlx_mts` stays off. Plan included
routing live translate through `engines/mlx_spec.py`.

**Already on `main` / PR #192:** `#183` wrapper + probe.
[`MAC_LATENCY.md`](https://github.com/wrbell/stark-translate/blob/09e4679a041fa9825d397436496d6f30cd49aa08/docs/archive/v2026.13/MAC_LATENCY.md)
§3: γ=1 acceptance 31.3%, medium 1339 vs 1393 ms greedy (**0.96×, not ≤0.85×**),
byte-identical 33/33, RoPE suspect rejected, default off. Canary ≥ 7/8 was not
published (greedy 8-slice historically 6/8).

**PR #192 live flag:** `dry_run_ab.py` still
`WARNING: MTS drafter load failed` and continues (`5154fb9` ~1258).

**`6935ecc` (not on PR yet):** `validate_live_mts` / `LIVE_MTS_UNAVAILABLE`
raises before any target load; `load_translation_models` raises if `USE_MTS`;
test `test_live_mts_rejects_before_any_target_model_load` asserts
`load_mlx_gemma` is not called.

**Do not confuse with prefix smoke.** Real-model **fixed-prefix cache**
equivalence (independent of MTP) is
`.cache/overnight-20260910/prefix-equivalence.json` at code
`7ef4b4c`: **35/36** cached vs uncached texts identical; miss is E4B
`canary_14` only. Purpose field: “development CPU activity excludes
performance promotion.” Tests:
`tests/test_overnight_prefix_cache.py` (also asserts prefix+MTP cannot
combine). This is **not** the MTP 0.85× gate and **not** 33/33 MTP identity.

**Close when:** explicit live reject is merged; comment that the speed gate
failed and MTP stays off. Do not claim 0.85×. Suggested comment:

> Closing #177 as a bounded experiment that missed the speed gate (medium
> 1339/1393 ms). `#183` probe + RoPE reject. `mlx_mts` default false. Live
> `--mts` now errors instead of swallowing loader failure (`6935ecc`).
> Prefix-cache 35/36 identity is a separate experiment, not this gate.

---

## #131 — Operator smoke

https://github.com/wrbell/stark-translate/issues/131

**Original Do:** fresh setup Mac and/or CUDA; operator preflight green;
`--dry-run-text` **plus one live mic utterance EN→ES and ES→EN**; W16 CT2
factory preference. **Done when:** operator UI loads, one partial + one final
on audience, no uncaught errors.

**File sessions on `5154fb9` (not live mic):**

| Session | Kind | Result |
|---|---|---|
| `20260909_233546_027169_en` | `replay`, input sha `6e89457f…` = 56.1 s verse-cue manifest | 8 CSV finals; **8/8** visible `stage=complete` browser ACKs; pause+resume 200; verses API 200; unapproved review note chunk 2 + subsequent GET; exit 0 |
| `20260909_233823_034893_es` | `replay` synthetic ES | 1 final; **1** visible complete ACK; lang_flip/stop 200; exit 0 |

**Mic session `20260909_233204_799019_en`:** `session_kind=live`,
`audio_source=mic`, 0 CSV rows. Lifecycle left **`status: running`** with no
`exit_code` / `ended_at` (UI can look running). Session log reaches
“Listening…” and does not contain the string `PortAudio`. Operator/root
report: stall in PortAudio input open. **User defers mic and physical tests
until tomorrow.** These file replays **do not** satisfy the live-mic Do item.

W16 CT2 still CUDA autodetection in tests; adapter absent here; Mac EN STT is
Parakeet. CUDA box unrun (`and/or` allows Mac-only setup).

**Keep open** until live mic EN and ES exist.

---

## #134 — Sunday dry-run / runbook

https://github.com/wrbell/stark-translate/issues/134

**Original Do:** walk the runbook on church hardware **or a laptop stand-in**;
time setup → first caption; operator-only vs helper-needed; **one full hymn +
one spoken segment**; file follow-ups for blocking UX. **Done when:** written
note (what worked, what broke, time-to-first-caption) and blocking UX holes
have their own issues.

Laptop stand-in **is** allowed. Live mic and church hardware are **not** in
the original body; do not add them as extra #134 gates.

**Current 56 s English input is not a full hymn.**
[`docs/evaluation/mac_v2026_14_verse_rehearsal.json`](evaluation/mac_v2026_14_verse_rehearsal.json):
56.106875 s, synthetic Piper cue (“Please turn to John chapter three verse
sixteen…”) + 45 s natural clip `stark_data/replay/mac_screening_45s.wav`.
Session `20260909_233546_027169_en` hashes to that file. Spoken segment:
**yes**. Full hymn: **no**. The older 68 s mix with a 15 s offset-600 excerpt
is also not a full hymn.

Observed on that laptop file session (artifacts, not a runbook write-up):
lifecycle start `2026-09-10T03:35:50Z`, first CSV final `2026-09-09T23:36:14`
local (~25 s later). That is not a documented setup→first-caption from a
cold runbook walk.

Root will run a full recorded church-service endurance/rehearsal later and
update evidence before closure.

**Keep open.**

---

## #132 — Multi-channel TTS

https://github.com/wrbell/stark-translate/issues/132

**Original Do** includes virtual cables. **Done when:** chosen device; EN and
ES independently; tests on the engine path.

**On `main` via [#188](https://github.com/wrbell/stark-translate/pull/188):**
code + `tests/test_tts_multichannel.py`. Physical second output / virtual
cable **not** evidenced. Builtin-speaker / WAV smoke is not independent
routing.

**Keep open.**

---

## #137 — Active learning / corrections

https://github.com/wrbell/stark-translate/issues/137

**Done when:** operator corrects a caption and that **pair** lands in a dated
corpus; retrain script documented even if the first retrain is a dry run.
**Do** also asks a recorded-Sunday correct → merge → smoke retrain.

Review UI is in PR #192, not `main`. Session
`stark_data/corrections/20260909_233546_027169_en.jsonl` revision 1:
`transcript_approved` and `translation_approved` **false**; note is rehearsal
draft persistence, not an approved pair. Merge/retrain of a real pair:
unperformed. Scripts documented.

**Keep open.**

---

## #135 — W16 + v2-cpo Mac A/B

https://github.com/wrbell/stark-translate/issues/135

No A/B note, no ship/no-ship with canary scores. Stock E4B default is not a
documented no-ship. W16 CT2 remains CUDA-only. **Keep open** (WSL).

---

## #136 — Jacobo preference triples

https://github.com/wrbell/stark-translate/issues/136

No 50–100 triples, no CPO continue, no COMET-22 vs v2-cpo. Default E4B still
does not emit Jacobo for “James wrote…”. Church prompt is a precursor, not
the AC. **Keep open.**

---

## #138 — Hindi zero-shot

https://github.com/wrbell/stark-translate/issues/138

**Original:** `target_lang_code="hi"` through the **live** pipeline; SOV
partial note; baseline on **church audio**.

Offline text probe
[`docs/evaluation/mac_v2026_14_hindi/`](evaluation/mac_v2026_14_hindi/README.md)
is not live `--lang hi` (`dry_run_ab.py` still `en`/`es` only) and not church
audio. **Keep open.**

---

## #133 — Live diarization

https://github.com/wrbell/stark-translate/issues/133

**On `main` via [#189](https://github.com/wrbell/stark-translate/pull/189):**
`--diarize` default off; rolling buffer; labels; unit tests. **Done when:**
two-speaker dry-run with distinct labels **without blowing p95**. Gate
unrun. **Keep open.**

---

## Gaps for root (no product edits here)

1. Push `6935ecc`…`53d8680` onto `codex/mac-reliability-roadmap` / PR #192
   before treating worker tests or live `--mts` reject as GitHub `main`
   evidence.
2. Live mic EN/ES tomorrow (#131). Do not close on file replay.
3. #134: capture a **full hymn** plus spoken; write setup→first-caption;
   church-service endurance is a later root update, not a new original
   requirement.
4. Prefix 35/36 ≠ MTP identity.

Issues remain OPEN on GitHub at this audit.
