# Backlog acceptance audit — September 10, 2026

All 43 canonical entries were compared with current source/evidence and live original GitHub issue text. This is an advisory snapshot; `docs/backlog.json` and GitHub issue states were not changed. The [machine-readable audit](backlog-audit.json) preserves every original acceptance, source hashes, original issue bodies, concrete next action and required receipt.

The Torch migration entry has a targeted follow-up after the 14:40–14:43 UTC full candidate replays; its previous finding is preserved in JSON. Other entries retain the original audit snapshot.

Snapshot: `c1a9041a48adc7c55e123eb39db02edb2a6fa6bd` plus the current follow-up working tree. PR #192 is merged; #134 and #176 are closed completed, while #177 is closed as a negative investigation. No publication, human/device gate or model-default promotion follows from those closures.

| Classification | Entries |
|---|---:|
| met | 7 |
| human or device | 5 |
| mixed | 12 |
| autonomous | 7 |
| external hardware | 7 |
| user decision | 3 |
| deferred user | 2 |

The remaining work is not entirely blocked on users or hardware. Local autonomy remains for the frozen EN/ES speed/quality experiments, current-artifact packaging checks, wider fallback-source classification, Mac dependency-source migration after successful isolated full-application validation, and documentation reconciliation. Source/widget/reliability acceptance also needs root reconciliation against already-recorded receipts.

New training preparation repairs the W17 module/config/initialization path and v2 corpus selection and supplies actual CPU trainer-data preflights. Sixty original unapproved Jacobo/Santiago candidates exist, with zero exact overlaps across eight available exclusion files; the required v2 holdout is absent. WSL inspection, approval, CPO continuation and quality gates remain pending.

Original-gate distinctions requiring root attention:

- #132 permits independent physical **or virtual** routing. The separate `physical-second-output` item requires two real outputs and audible verification; do not silently rewrite the original issue to exclude a valid virtual-routing demonstration.
- #133 asks that speaker labels preserve p95 latency. The canonical +50 ms guard is additional local policy and must be labeled that way.
- #134 expressly permits a laptop stand-in and is already closed. Microphone and physical-display gates remain separate.
- #137 permits the first retrain to be a dry run, but a merger dry run alone is not that step. A real approved Sunday correction is still absent.
- #174/#175 closure reflects source/documentation fixes; the broader CUDA experiments still require target hardware.
- Torch 2.13/TorchAudio 2.11 now passes full isolated application install/import/VAD/audit and real normalized EN/ES replay. The [candidate receipts and bounded migration recommendation](torch-full-application-candidate.md) supersede the earlier same-version audio-wheel failure. Production source/runtime promotion remains; no service or performance certification follows.
- Public EN/ES read-speech annotations enable separate engineering comparisons; they do not supply locally approved church references, speaker labels or microphone evidence.

| Item | Classification | Next action |
|---|---|---|
| `pr-192-integration` | met | Retain the immutable PR192 receipt; handle this follow-up branch under its own review and validation identity. |
| `mac-live-mic-stall` | human or device | At the user-deferred attended session, run built-in capture through operator and audience pages and provoke a stalled/disconnected input. |
| `issue-131-smoke` | human or device | Run attended live EN and ES microphone utterances; inspect the configured faster-whisper adapter preference separately from Mac auto EN policy. |
| `issue-134-sunday-dry-run` | met | Keep #134 closed; continue #193/#194 under their own original acceptance. |
| `issue-132-tts-routing` | mixed | Root should distinguish original virtual-or-physical independent routing acceptance from the separate physical-second-output gate; demonstrate selected independent routes when available. |
| `physical-second-output` | human or device | At the attended session, route each language to a distinct real output, restart, unplug/replug and verify audible playback. |
| `issue-133-diarize-gate` | mixed | Obtain a labeled natural two-speaker clip, then run the opt-in on/off comparison and offline-path check with one model process at a time. |
| `natural-two-speaker` | human or device | Obtain a natural church two-speaker segment and human-marked transitions, retaining uncertain regions. |
| `natural-spanish-refs` | mixed | Use the audited public set for separate engineering evidence now; obtain and annotate church/natural utterances with local human approval for this gate. |
| `bilingual-blinded-review` | human or device | Supply the existing blinded form to a bilingual reviewer while keeping the key separate; retain every decision. |
| `visible-browser-timing-run` | mixed | Finish separate schema-2 replay evidence; when the user-deferred unlocked/attended condition is available, record the actual audience display visibility. |
| `caption-delivery-goal` | autonomous | Complete the frozen new EN/ES development protocol, preserve negative arms, freeze a selector before untouched confirmation, and compare schema-2 latency with quality guards. |
| `overnight-latency-scheduling` | met | Preserve the rejected results and unchanged defaults; permit only a distinct explicitly frozen hypothesis. |
| `lite-cpu-inference` | mixed | Finish separate CPU development/confirmation evidence and inspect sparse previews; run setup/doctor and EN/ES on representative no-GPU x86 hardware when available. |
| `rtx2070-native-validation` | external hardware | On a 2070-class native Windows host, run setup/doctor/operator with lite-cuda-8gb and EN/ES replay. |
| `overnight-operator-ui` | autonomous | Root should reconcile the current runbook, widget sources, Tidy/QR receipts and UI evidence against this bounded acceptance; retain unrelated device/quality gates. |
| `overnight-reliability` | autonomous | Refresh the stale Lite next action and root-review source/test acceptance; preserve the distinct real-microphone item as pending. |
| `issue-135-mac-ab` | mixed | Locate/transfer W16 CT2 and v2-cpo artifacts, verify them, then perform the specified Mac live A/B and terminology/health checks. |
| `issue-136-jacobo-cpo` | mixed | On WSL, audit against required v2 holdouts, obtain bilingual review, export a separate approved/scored pool, continue from v2-cpo and re-score eight canaries/500 verses. |
| `issue-137-active-learning` | mixed | After human approval, export one recorded-Sunday correction, merge into a scratch dated training corpus, then execute the appropriate actual recipe CPU preflight/dry-run. |
| `issue-138-hindi-zero-shot` | user decision | Await a later explicit Hindi R&D decision; preserve the offline baseline without claiming live acceptance. |
| `issue-176-multiprocess` | met | Keep closure and maintain prompt/stop contracts during related edits; no new multiprocess performance claim. |
| `issue-177-mtp` | deferred user | Keep MTP off; resume only for a distinct later hypothesis with all original promotion gates. |
| `conservative-marian-routing` | mixed | Keep routing opt-in; compare allowlisted phrase routing using independent natural references, then obtain bilingual quality review. |
| `mac-defaults-frozen` | met | Keep defaults frozen until a distinct matched improvement and separate quality review meet promotion criteria. |
| `mac-reliability-implementation` | met | Retain merged-source evidence and work on the separate device/human gates. |
| `mac-cpu-test-suite` | met | Keep frozen counts unchanged; bind each follow-up test/static/model receipt to its own source and scope. |
| `packaging-artifacts-local` | autonomous | After source stabilizes, rebuild and compare artifact contents, install outside checkout, run required EN/ES smoke under serial scheduling and record hashes. |
| `pypi-publication` | user decision | Await the separate publication decision and trusted-publisher mapping. |
| `wsl-phase4` | external hardware | On WSL storage execute runbook §1 using the real sermon corpus, preserving cutoff/holdout splits. |
| `wsl-e4b-domain-sft` | external hardware | After real corpus availability, run the Gemma recipe preflight on WSL then domain SFT and GGUF export/sanity. |
| `wsl-w17-export` | external hardware | Run real WSL W16/config/corpus preflight, then mine/align/train/export and compare with W16 under the original benchmark. |
| `cuda-latency-proposal` | external hardware | On A2000 WSL execute the proposal build, MTP/flash-attention and W16/Parakeet probes; keep release uploads separately authorized. |
| `v2026-9-followups` | external hardware | When WSL is available, execute each remaining hypothesis under proposal gates; retain negative results. |
| `security-b615-pinning` | autonomous | Root-review live fallback changes and document justified operator-only remaining paths; rerun the explicit wider inventory without claiming B615 globally clear. |
| `multilingual-expansion` | user decision | Await explicit expansion scope/priority; do not train or integrate languages as an EN/ES side effect. |
| `macos-shortcuts` | deferred user | No action until adopted by the user. |
| `docs-refresh-remaining` | autonomous | Root reconcile this 43-item audit into current guides/backlog, then run renderer validation, link checks and documentation tests. |
| `mac-torch-security-migration` | autonomous | Review Mac-only dependency metadata and the complete tested constraint file, then validate a fresh final-source runtime before changing a launcher; preserve stt_env and non-Mac/CUDA constraints. Full isolated candidate EN/ES/audit evidence is complete. |
| `windows-msi-bootstrap` | external hardware | On a clean Windows account with matching package availability, exercise install, Start Menu launch, setup/models, EN/ES, offline relaunch and uninstall. |
| `wsl-training-recipe-checks` | mixed | Refresh stale source-defect notes; on WSL run both --dry-run recipes with actual W16/config/v2 corpus and retain the resulting CPU reports. |
| `hymn-translation-boundary` | mixed | Keep the raw regression input unchanged, obtain independent boundary/bilingual review, then compare bounded raw/delimited input with identical models and semantic/latency controls. |
| `hymn-capture-suppression` | mixed | Label a bounded hymn→speech transition with uncertain regions, then compare candidate detector/scheduling with unchanged controls including quiet prayer/short EN/ES replies. |
