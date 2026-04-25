"""Operator control plane (Phase 9).

FastAPI app + vanilla-JS frontend that lets a non-technical operator drive
the live pipeline without touching the CLI. Replaces the developer-grade
5-tab workflow shipped pre-Phase-9.

Entry point: ``operator_app.main:app`` — run via ``uvicorn operator_app.main:app``.

The package is named ``operator_app`` because Python's stdlib has an
``operator`` module (``operator.itemgetter`` etc.); using ``operator``
as a package name causes import collisions.
"""
