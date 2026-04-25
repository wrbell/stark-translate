# macOS native `.app` — deferred to v2026.8

> **Status:** deferred. The Apple Developer enrollment ($99/yr) is the
> long-lead gate; until that's set up, Mac users install via the
> [`pypi.md`](./pypi.md) path:
>
> ```bash
> uv tool install 'stark-translate[mlx]'
> stark-translate setup
> stark-translate operator
> ```
>
> This works today for technically-comfortable Mac users.

A native `.app` (drag-to-Applications, double-click, browser opens) is the
better UX for non-technical Mac volunteers, but requires Apple Developer
enrollment, notarization, and Sparkle for auto-update. The full design
notes from the v2026.7 packaging research are preserved at
`~/.claude/plans/we-haven-t-worked-on-lexical-moth-agent-ab3712d4ca033d935.md`
and will be lifted directly when v2026.8 reactivates this track.

---

## What's planned (v2026.8)

| Component | Tool |
|---|---|
| Wrapper | [Briefcase 0.3.25](https://briefcase.readthedocs.io/) |
| Auto-updater | [Sparkle](https://sparkle-project.org/) |
| CI runner | self-hosted M-series (per user decision) |
| Target arch | arm64 only (no Intel Macs) |
| Mic entitlement | `NSMicrophoneUsageDescription` |
| Appcast hosting | `wrbell.github.io/stark-translate/appcast.xml` (free GitHub Pages) |

The `.app` will subprocess `run_operator.sh` exactly as the Linux Docker
path does — there's no separate code path for Mac vs Linux at runtime.

---

## Why not Apple Container framework?

The macOS 26 `container` framework added in 2025 doesn't pass Metal/MLX
through to containers (apple/container#62). For an MLX-based inference app
that needs the GPU, native `.app` is the only path. We considered shipping
a Mac container path for parity with Linux; it's not viable.

---

## See also

- [`pypi.md`](./pypi.md) — current Mac install path until v2026.8
- [`../../CLAUDE-macbook.md`](../../CLAUDE-macbook.md) — Mac dev environment notes
