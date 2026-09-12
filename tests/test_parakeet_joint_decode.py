"""Production joint decode fails closed and restores the original method."""

import logging
from unittest.mock import Mock

import pytest

from engines import parakeet_joint_decode as joint
from engines.parakeet_mlx_engine import ParakeetMLXEngine


class Model:
    def decode_greedy(self):
        return "stock"


def test_source_hash_mismatch_preserves_stock(caplog):
    original = Model.decode_greedy
    with caplog.at_level(logging.WARNING):
        assert joint.install_qualified_joint_decode(Model()) is None
    assert Model.decode_greedy is original
    assert len(caplog.records) == 1
    assert "keeping stock decode" in caplog.text


def test_unavailable_source_preserves_stock(monkeypatch, caplog):
    monkeypatch.setattr(joint.inspect, "getsource", Mock(side_effect=OSError("source unavailable")))
    original = Model.decode_greedy
    assert joint.install_qualified_joint_decode(Model()) is None
    assert Model.decode_greedy is original
    assert len(caplog.records) == 1


@pytest.mark.parametrize("matching_ast", [True, False])
def test_qualified_install_checks_ast_and_restores(monkeypatch, matching_ast):
    def marker(self):
        return "joint"

    build = Mock(
        return_value=(
            marker,
            {"transformed_ast_sha256": joint.QUALIFIED_TRANSFORMED_AST_SHA256 if matching_ast else "changed"},
        )
    )
    monkeypatch.setattr(joint, "joint_method", build)
    model = Model()
    original = Model.decode_greedy
    restore = joint.install_qualified_joint_decode(model)
    build.assert_called_once_with(original, joint.QUALIFIED_SOURCE_SHA256)
    if matching_ast:
        try:
            assert model.decode_greedy() == "joint"
            assert restore is not None
        finally:
            restore()
        restore()  # Repeated cleanup is harmless.
    else:
        assert restore is None
    assert Model.decode_greedy is original


@pytest.mark.parametrize("disabled", ["0", "false", "off"])
def test_env_disables_engine_installer(monkeypatch, disabled):
    import parakeet_mlx

    from engines import parakeet_mlx_engine as engine_module

    monkeypatch.setenv("STARK_PARAKEET_JOINT_EVAL", disabled)
    monkeypatch.setattr("engines.model_paths.resolve_model_for_loading", lambda _: "/installed/fake")
    monkeypatch.setattr(parakeet_mlx, "from_pretrained", Mock(return_value=Model()))
    monkeypatch.setattr("engines.mlx_engine.materialize_mlx_model", Mock())
    installer = Mock()
    monkeypatch.setattr(engine_module, "install_qualified_joint_decode", installer)
    engine = ParakeetMLXEngine(warmup_seconds=0)
    engine.load()
    installer.assert_not_called()
    assert not engine.joint_scalar_eval_active
    engine.unload()
