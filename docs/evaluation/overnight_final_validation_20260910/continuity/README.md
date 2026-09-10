# Final built wheel continuity: 752ab9a → 84832fb

**138 non-Markdown runtime code/resource members are byte-identical; exactly five operator lifecycle modules changed as expected.** Both wheels have the same 152-member runtime inventory: 143 non-Markdown members and 9 Markdown files. No runtime paths were added or removed. There are no unexamined runtime members.

All inference engine/pipeline code and other resources outside the five named operator modules match 752ab9a exactly. The successful Standard and Lite hours remain bound to their actual 752ab9a installation. This companion establishes the exact final code difference; it does not invent another hour or promote a latency, quality, physical-device or target-hardware gate.

| Artifact | Source | SHA-256 |
|---|---|---|
| Endurance canonical wheel | `752ab9a351815feee4b8cd155f732c588cb30a6c` | `7477574d25c91739b6a88ca142a35bf36258599a66671b8dbb32237d1fa852b5` |
| Final canonical wheel | `84832fb98caf4504cf65b1f5969558b0be911f5c` | `ac3d5216ce79f2ef004e2ed5be4c42d5eccb3f2ce8612661a2a7d138c7c2c41e` |

Each canonical wheel independently matches its sdist-rebuilt and Mac-ZIP-rebuilt wheel. All 152 actual runtime member hashes match the respective build manifests. Container hashes differ because operator source, Markdown and wheel metadata changed.

## Disclosed operator cleanup delta

These five members are excluded only from the identical subset. All five are fully compared and hash-mapped below. They stop existing application workers on lifespan exit, preserve exception cleanup, avoid constructing absent workers and report failed bounded joins. They do not change inference engines, model selection, caption generation, audio processing or pipeline timing.

| Wheel member | 752 SHA-256 | 848 SHA-256 |
|---|---|---|
| `operator_app/audio.py` | `37566c2f55eb22cf4ddc799c3410956d531e1f18ab1191e50afceb7344d229f7` | `82ffab0935fe07fc622dba6abb7633dc74b1592b9ac5158a6847be27f37c1875` |
| `operator_app/features.py` | `831777725e90f7969ccfad8550bb70a04da388e06001c1e37e5fd160fcf12d09` | `56d651aa5b56420237ef358f6e79f934e5c6f8c5ffdcb1c4b8680f48e4d53ac0` |
| `operator_app/main.py` | `f7644063151647b352928cab1cfa2fe140409c9292bb0e6ab085bc201b901a83` | `ace170114555b3332ca154b47d855eccc2c1f907fe703f97abdb5293bf930bef` |
| `operator_app/metrics.py` | `496cd718f026e46c8dbdd317b087960c2912dacde3030abd3c25a1206629d52c` | `cb0b35dae6c0aff85e6c3982bc54569d54d2ec31974d6f80d92e32df6c235642` |
| `operator_app/pipeline_manager.py` | `519c474c26804703370a6bdf9d6fc28c802bd820eba7a326256b4073bf62b1bb` | `74920f76011d361107b1eff63442693d333875d0c0df1be0397b6309b528e290` |

## Markdown and metadata

Changed Markdown: `displays/CLAUDE.md`, `engines/AGENTS.md`, `tools/CLAUDE.md`.
Unchanged Markdown: `displays/AGENTS.md`, `displays/operator/README.md`, `engines/CLAUDE.md`, `features/AGENTS.md`, `features/CLAUDE.md`, `tools/AGENTS.md`.
Changed dist-info members: `stark_translate-2026.14.0.0.dist-info/METADATA`, `stark_translate-2026.14.0.0.dist-info/RECORD`.
Parsed package metadata headers, including requirements/version, are identical. Exact old/new hashes and sizes for every one of the 157 wheel members, both build reports and all six wheel containers are in `report.json`.

## Validation boundary

The final 84832fb artifact receipt passed. It records six actual native sounddevice initialization/rapid-process-exit checks, with no model imports and no remaining polling workers after lifespan. Zero inputs/outputs were visible in that execution context; this does not certify microphone or speaker hardware. The earlier 60ad4db shutdown failure remains retained separately; this final receipt does not rewrite it.

This comparison itself only read already-built files. No extraction, package import/execution, model, device, test, build, tracked source, installed environment or Git/index mutation was performed. Every inference resource is compared, but model weight files are external to the wheel and are not newly revalidated by this member comparison.
