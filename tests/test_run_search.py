from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch
from syntheseus import Bag, Molecule, SingleProductReaction

from retrochimera.inference import smiles_transformer as smiles_transformer_inference
from retrochimera.inference.retrochimera import RetroChimeraModel
from retrochimera.inference.smiles_transformer import AbstractSmilesTransformerModel


def test_parallel_ensemble_workers_disable_gradients(monkeypatch) -> None:
    class Stream:
        def synchronize(self) -> None:
            pass

    class Submodel:
        grad_enabled = True

        def __call__(self, inputs, num_results):
            self.grad_enabled = torch.is_grad_enabled()
            return [
                [
                    SingleProductReaction(
                        product=input,
                        reactants=Bag([Molecule("C")]),
                        metadata={"probability": 1.0},
                    )
                ]
                for input in inputs
            ]

    stream = Stream()
    submodel = Submodel()
    executor = ThreadPoolExecutor(max_workers=1)
    ensemble: Any = object.__new__(RetroChimeraModel)
    ensemble._cached_streams = [stream]
    ensemble._cached_executor = executor
    ensemble._models = [submodel]
    ensemble._model_weights = [[1.0]]
    ensemble._model_names = ["submodel"]
    ensemble.probability_from_score_temperature = 8.0
    ensemble.consensus_only = False
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())

    try:
        ensemble._get_reactions([Molecule("CC")], num_results=1)
    finally:
        executor.shutdown()

    assert not submodel.grad_enabled


def test_compute_probs_disables_gradients() -> None:
    class Model:
        def compute_probs(self, reaction_smiles, minibatch_size):
            assert not torch.is_grad_enabled()
            return [0.5], [0.5]

    wrapper: Any = object.__new__(AbstractSmilesTransformerModel)  # type: ignore[type-abstract]
    wrapper.model = Model()

    assert wrapper.compute_probs(["C>>C"], minibatch_size=1) == ([0.5], [0.5])


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
