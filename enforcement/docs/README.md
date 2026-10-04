# Docs structure

The docs structure domain owns this folder.

`readme.markdownlint-cli2.jsonc` sets markdownlint MD043 for a README.
The title is one heading.
`## License` is the last heading.

`adr.markdownlint-cli2.jsonc` sets MD043 for a MADR 4.0.0 record.
The docs workflow passes that file as its config.
Do not copy it into `docs/decisions/`.
A nested config would replace the root Markdown config.
`MD041` is off because a record starts with front matter.

`tools/check_docs.py` checks the README, decision-record names and
front matter, a root `CITATION.cff`, and every `CITATION.cff` under
`templates/`.
The job `docs-lint` runs when `tools/check_docs.py` exists.
`ci.yml` calls the workflow.

Both configs ignore `.agents/**` and `third_party/**`.
