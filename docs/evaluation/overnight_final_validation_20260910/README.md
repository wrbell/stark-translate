# Retained final integration evidence

These are exact copies of finalized CPU/static/artifact/CI/continuity receipts and
selected modest logs for source `84832fb`, plus the failed `60ad4db` artifact
receipt and its short process log. See the [consolidated report](../overnight_final_validation_20260910.md).

[inventory.json](inventory.json) maps every copied file to its original local path,
size and SHA-256; [SHA256SUMS](SHA256SUMS) verifies retained bytes with
`shasum -a256 -c SHA256SUMS` from this directory. The continuity comparator is kept
as `continuity/compare_wheels.py.txt`, an inert exact-byte copy of the original
script; its original filename is recorded in the inventory. Its companion
manifest retains the original source filenames. Do not interpret it as a new run.

No OS crash report/core dump, audio, model, private correction or environment-variable
dump is included. Local paths, package names and command arguments remain as
provenance. Credential-pattern review was heuristic, not a privacy certification.
Original 752 receipts and endurance archives were not rewritten.
