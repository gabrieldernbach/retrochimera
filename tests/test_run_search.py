from types import SimpleNamespace
from typing import Any

from retrochimera.inference import smiles_transformer as smiles_transformer_inference
from retrochimera.inference.smiles_transformer import AbstractSmilesTransformerModel


def test_canonicalization_pool_is_reused(monkeypatch) -> None:
    pool = SimpleNamespace()
    calls = 0

    def create_pool(max_workers):
        nonlocal calls
        calls += 1
        assert max_workers == 3
        return pool

    model: Any = object.__new__(AbstractSmilesTransformerModel)  # type: ignore[type-abstract]
    model._canonicalization_processes = 3
    model._canonicalization_pool = None
    monkeypatch.setattr(smiles_transformer_inference, "_get_reusable_executor", create_pool)

    assert model._get_canonicalization_pool() is pool
    assert model._get_canonicalization_pool() is pool
    assert calls == 1
