"""The actual parser declaration provides an explicit negative TTS override."""

import argparse
import ast
from pathlib import Path


def test_tts_parser_supports_an_explicit_off_without_loading_runtime():
    source = Path(__file__).resolve().parents[1] / "dry_run_ab.py"
    tree = ast.parse(source.read_text())
    declaration = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--tts"
    )
    parser = argparse.ArgumentParser()
    eval(compile(ast.Expression(declaration), str(source), "eval"), {"parser": parser, "argparse": argparse})
    assert parser.parse_args([]).tts is False
    assert parser.parse_args(["--tts"]).tts is True
    assert parser.parse_args(["--tts", "--no-tts"]).tts is False
