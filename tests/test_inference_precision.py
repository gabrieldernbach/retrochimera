from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from syntheseus import Molecule

from retrochimera.inference import smiles_transformer as inference
from retrochimera.inference.smiles_transformer import SmilesTransformerModel
from retrochimera.inference.smiles_transformer_forward import SmilesTransformerForwardModel


@pytest.fixture(params=[SmilesTransformerModel, SmilesTransformerForwardModel])
def make_model(request, tmp_path, monkeypatch):
    # Exercise the real wrapper constructors without a checkpoint or CUDA allocation.
    (tmp_path / "model.ckpt").touch()
    (tmp_path / "vocab.txt").touch()
    model = SimpleNamespace(
        max_seq_len=512,
        tokenizer=SimpleNamespace(log_vocab_info=lambda: None),
        eval=lambda: None,
    )
    model.to = lambda device: model
    monkeypatch.setattr(inference.TransformerModel, "load_from_checkpoint", lambda *a, **k: model)

    def create(**kwargs):
        return request.param(model_dir=tmp_path, **kwargs)

    return create


@pytest.mark.parametrize(
    "precision,expected_dtype",
    [
        ("float32", None),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_cuda_precision_selection(make_model, monkeypatch, precision, expected_dtype):
    cuda_probe = Mock(side_effect=AssertionError("Explicit precision must not query CUDA support"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", cuda_probe)
    wrapper = make_model(device="cuda:3", inference_precision=precision)

    assert wrapper._autocast_dtype == expected_dtype
    cuda_probe.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_default_precision_is_float32(make_model, monkeypatch, device):
    cuda_probe = Mock(side_effect=AssertionError("Default must not query CUDA support"))
    autocast = Mock(side_effect=AssertionError("Default must not add autocast"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", cuda_probe)
    monkeypatch.setattr(torch, "autocast", autocast)

    wrapper = make_model(device=device)
    assert wrapper._autocast_dtype is None
    with wrapper._autocast_context():
        pass

    cuda_probe.assert_not_called()
    autocast.assert_not_called()


def test_explicit_cuda_bf16_defers_validation_to_autocast(make_model, monkeypatch):
    cuda_probe = Mock(side_effect=AssertionError("Explicit precision must not query CUDA support"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", cuda_probe)
    wrapper = make_model(device="cuda:0", inference_precision="bfloat16")
    cuda_probe.assert_not_called()

    autocast = Mock(side_effect=RuntimeError("Current CUDA Device does not support bfloat16"))
    monkeypatch.setattr(torch, "autocast", autocast)
    with pytest.raises(RuntimeError, match="does not support bfloat16"):
        wrapper._autocast_context()
    autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("precision", ["auto", "float64"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_invalid_precision_is_rejected(make_model, precision, device):
    with pytest.raises(ValueError, match=f"Unsupported inference_precision {precision}"):
        make_model(device=device, inference_precision=precision)


@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
def test_cpu_does_not_add_autocast(make_model, monkeypatch, precision):
    cuda_probe = Mock(side_effect=AssertionError("CPU must not query CUDA support"))
    autocast = Mock(side_effect=AssertionError("CPU must not add autocast"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", cuda_probe)
    monkeypatch.setattr(torch, "autocast", autocast)
    wrapper = make_model(device="cpu", inference_precision=precision)

    with wrapper._autocast_context():
        pass

    cuda_probe.assert_not_called()
    autocast.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_autocast_uses_selected_dtype(make_model, monkeypatch, dtype):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    precision = "float16" if dtype == torch.float16 else "bfloat16"
    wrapper = make_model(device="cuda:0", inference_precision=precision)
    autocast = Mock(side_effect=lambda **kwargs: nullcontext())
    monkeypatch.setattr(torch, "autocast", autocast)

    with wrapper._autocast_context():
        pass

    autocast.assert_called_once_with(device_type="cuda", dtype=dtype)


def test_float32_does_not_add_autocast(make_model, monkeypatch):
    wrapper = make_model(device="cuda:0", inference_precision="float32")
    autocast = Mock(side_effect=AssertionError("float32 must not add autocast"))
    monkeypatch.setattr(torch, "autocast", autocast)

    with wrapper._autocast_context():
        pass

    autocast.assert_not_called()


def test_float32_preserves_enclosing_cuda_autocast(make_model, monkeypatch):
    # CUDA autocast thread-local state can be tested without allocating a CUDA tensor.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    wrapper = make_model(device="cuda:0", inference_precision="float32")
    original_enabled = torch.is_autocast_enabled()
    original_dtype = torch.get_autocast_gpu_dtype()
    original_matmul_precision = torch.get_float32_matmul_precision()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with wrapper._autocast_context():
            assert torch.is_autocast_enabled()
            assert torch.get_autocast_gpu_dtype() == torch.float16
        assert torch.is_autocast_enabled()

    assert torch.is_autocast_enabled() == original_enabled
    assert torch.get_autocast_gpu_dtype() == original_dtype
    assert torch.get_float32_matmul_precision() == original_matmul_precision


def test_compute_probs_enters_autocast_and_inference_mode(make_model, monkeypatch):
    wrapper = make_model(device="cpu")
    active = []

    @contextmanager
    def autocast_context():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def compute_probs(reaction_smiles, minibatch_size):
        assert active == [True]
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        assert reaction_smiles == ["C>>C"]
        assert minibatch_size == 1
        return [0.5], [0.5]

    monkeypatch.setattr(wrapper, "_autocast_context", autocast_context)
    wrapper.model.compute_probs = compute_probs
    original_inference_mode = torch.is_inference_mode_enabled()

    assert wrapper.compute_probs(["C>>C"], minibatch_size=1) == ([0.5], [0.5])
    assert not active
    assert torch.is_inference_mode_enabled() == original_inference_mode


def test_decoding_enters_autocast(monkeypatch, tmp_path):
    # Run the backward wrapper's normal invocation through decoding and postprocessing.
    # The shared decoding implementation also serves the forward wrapper.
    (tmp_path / "model.ckpt").touch()
    (tmp_path / "vocab.txt").touch()
    tokenizer = SimpleNamespace(
        log_vocab_info=lambda: None,
        encode=lambda smiles, **kwargs: ([[1, 2]], [[False, False]]),
        decode=lambda token_ids: ["C"],
    )
    model = SimpleNamespace(max_seq_len=512, tokenizer=tokenizer, eval=lambda: None)
    model.to = lambda device: model
    monkeypatch.setattr(inference.TransformerModel, "load_from_checkpoint", lambda *a, **k: model)
    wrapper = SmilesTransformerModel(
        model_dir=tmp_path, device="cpu", beam_size=1, augmentation_size=1, use_cache=False
    )
    active = []
    calls = []

    @contextmanager
    def autocast_context():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def translate_batch(batch, attn_debug):
        assert active == [True]
        assert batch["batch_size"] == 1
        assert not attn_debug
        calls.append(batch)
        return {"predictions": [[torch.tensor([1])]], "scores": [[0.0]]}

    def map_predictions(fn, lines):
        assert not active
        return map(fn, lines)

    monkeypatch.setattr(wrapper, "_autocast_context", autocast_context)
    monkeypatch.setattr(
        inference, "Translator", lambda **kwargs: SimpleNamespace(translate_batch=translate_batch)
    )
    monkeypatch.setattr(
        inference, "get_reusable_executor", lambda **kwargs: SimpleNamespace(map=map_predictions)
    )
    molecule = Molecule("CC")

    predictions = wrapper([molecule], num_results=1)

    assert len(calls) == 1
    assert not active
    assert len(predictions) == len(predictions[0]) == 1
    assert predictions[0][0].product == molecule
    assert [mol.smiles for mol in predictions[0][0].reactants] == ["C"]
