"""Hash-qualified, output-exact joint scalar materialization for Parakeet."""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import textwrap
from collections.abc import Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)
QUALIFIED_SOURCE_SHA256 = "ce20ad969c6fd7aba0168f06356bf167f0334f590c9f4f33cc47ba46b8afa79d"
QUALIFIED_TRANSFORMED_AST_SHA256 = "754a54566f81fb336936746a0c4a876c5a2466d4cdcd1252dfdf862f8b120a79"

SCALARS = ("pred_token", "confidence", "decision")


def joint_method(original, expected_source_sha256):
    """Clone a trusted installed method after verifying the profiled source hash."""
    if original.__closure__:
        raise ValueError("Expected non-closure installed greedy method")
    source = textwrap.dedent(inspect.getsource(original))
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if source_hash != expected_source_sha256:
        raise ValueError("Installed greedy source differs from qualifying profile")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("Expected one undecorated installed function")
    if any(isinstance(node, ast.Name) and node.id.startswith("_stark_joint_") for node in ast.walk(tree)):
        raise ValueError("Reserved temporary name already present")
    assignments: dict[str, list[ast.Assign]] = {name: [] for name in SCALARS}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in assignments
        ):
            assignments[node.targets[0].id].append(node)
    if any(len(nodes) != 1 for nodes in assignments.values()):
        raise ValueError("Scalar assignment count changed")
    scalar_nodes = [assignments[name][0] for name in SCALARS]
    blocks = [
        node.body
        for node in ast.walk(tree)
        if isinstance(node, ast.While) and all(scalar in node.body for scalar in scalar_nodes)
    ]
    if len(blocks) != 1:
        raise ValueError("Scalar assignments must share one greedy while body")
    block = blocks[0]
    indices = [block.index(node) for node in scalar_nodes]
    if indices != sorted(indices):
        raise ValueError("Scalar assignment order changed")
    # Delay only scalar materialization: intermediary graph expressions cannot
    # depend on a converted scalar or have branch/side-effect statements.
    for node in block[indices[0] + 1 : indices[-1]]:
        if node in scalar_nodes:
            continue
        if not isinstance(node, ast.Assign) or any(isinstance(n, ast.Name) and n.id in SCALARS for n in ast.walk(node)):
            raise ValueError("Scalar dependency/order changed inside sampling block")
    for name, node in zip(SCALARS, scalar_nodes):
        converter = "float" if name == "confidence" else "int"
        value = node.value
        if (
            not isinstance(value, ast.Call)
            or ast.unparse(value.func) != converter
            or len(value.args) != 1
            or value.keywords
        ):
            raise ValueError("Scalar converter changed: " + name)
        if name != "confidence" and (
            not isinstance(value.args[0], ast.Call) or ast.unparse(value.args[0].func) != "mx.argmax"
        ):
            raise ValueError("Argmax expression changed: " + name)
        node.targets = [ast.Name(id="_stark_joint_" + name, ctx=ast.Store())]
        node.value = value.args[0]  # exact original expression, no dtype/math changes
    materialize = ast.Expr(
        ast.Call(
            ast.Attribute(ast.Name("mx", ast.Load()), "eval", ast.Load()),
            [ast.Name("_stark_joint_" + name, ast.Load()) for name in SCALARS],
            [],
        )
    )
    conversions = [
        ast.Assign(
            [ast.Name(name, ast.Store())],
            ast.Call(
                ast.Name("float" if name == "confidence" else "int", ast.Load()),
                [ast.Name("_stark_joint_" + name, ast.Load())],
                [],
            ),
        )
        for name in SCALARS
    ]
    block[indices[-1] + 1 : indices[-1] + 1] = [materialize, *conversions]
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    # Source is inspect.getsource of the imported method, hash-bound to a
    # qualifying profiler receipt and AST-checked above; never supplied code.
    exec(compile(tree, original.__code__.co_filename + ":stark-joint-eval", "exec"), namespace)  # nosec B102
    cloned = namespace[original.__name__]
    cloned.__defaults__, cloned.__kwdefaults__ = original.__defaults__, original.__kwdefaults__
    return cloned, {
        "function": original.__qualname__,
        "original_source_sha256": source_hash,
        "transformed_ast_sha256": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        "original_source_file": original.__code__.co_filename,
        "materialization": "mx.eval(token_argmax, unchanged_entropy_confidence, duration_argmax) once per greedy step before same int/float reads",
    }


@contextmanager
def install_joint_method(model, method):
    cls = type(model)
    previous = cls.__dict__.get("decode_greedy")
    try:
        cls.decode_greedy = method
        yield
    finally:
        if previous is None:
            delattr(cls, "decode_greedy")
        else:
            cls.decode_greedy = previous


def install_qualified_joint_decode(model) -> Callable[[], None] | None:
    """Install only the qualified transform; retain stock decoding on mismatch."""
    try:
        method, identity = joint_method(type(model).decode_greedy, QUALIFIED_SOURCE_SHA256)
        if identity["transformed_ast_sha256"] != QUALIFIED_TRANSFORMED_AST_SHA256:
            raise ValueError("Transformed greedy AST differs from qualifying profile")
        installation = install_joint_method(model, method)
        installation.__enter__()
    except (ValueError, TypeError, OSError, AttributeError, KeyError) as exc:
        logger.warning("Parakeet joint scalar decode unavailable; keeping stock decode: %s", exc)
        return None

    def restore() -> None:
        installation.__exit__(None, None, None)

    return restore
