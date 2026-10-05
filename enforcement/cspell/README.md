# cspell

The Markdown domain owns this folder.
Copy `cspell.json` into a repository that wants the advisory spell check.
The job does not block a merge.
It ignores `.agents/`, `third_party/`, and downloaded Vale packages.

`words/engineering.txt` accepts project terms such as OpenFOAM, Ansys,
SolidWorks, Reynolds, Nusselt, and k-omega.

Do not auto-edit graded senior design report text.
The check only reports.
