"""Briefcase MSI compat shim.

The Briefcase Windows packager (v0.3.x) derives a Python module name from the
app key in ``pyproject.toml`` — ``stark-translate`` becomes ``stark_translate``
— and asserts that module appears in the ``sources`` list, even when
``external_package_path`` is set. We use external-package mode (the MSI ships
the PyApp launcher, not a Briefcase-built Python stub), so this package is a
no-op at runtime; it exists only to satisfy that check.

The real entry points live in ``operator_app`` (CLI + FastAPI control plane).
``operator_app.cli:main`` is the CLI dispatcher, and ``stark-translate`` is
already wired to it via ``[project.scripts]`` in pyproject.toml.
"""

# Re-export the CLI main so `from stark_translate import main` works as an
# alias if anything looks for it.
from operator_app.cli import main as main

__all__ = ["main"]
