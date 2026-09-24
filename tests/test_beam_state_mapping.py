from types import SimpleNamespace
from typing import Any

import pytest
import torch

from retrochimera.opennmt.decode.beam_search import BeamSearch
from retrochimera.opennmt.decode.translator import Translator
from retrochimera.opennmt.modules.average_attention import AverageAttention
from retrochimera.opennmt.modules.transformer_decoder import TransformerDecoder


def _decoder(self_attn_type: str = "scaled-dot") -> TransformerDecoder:
    decoder = TransformerDecoder(
        num_layers=1,
        d_model=4,
        heads=2,
        d_ff=8,
        self_attn_type=self_attn_type,
        dropout=0.0,
        attention_dropout=0.0,
    )
    decoder.eval()
    decoder.init_state(torch.tensor([[[10], [10], [20], [20]]]), None, None)
    return decoder


@pytest.mark.parametrize("self_attn_type", ["scaled-dot", "average"])
def test_map_state_can_reorder_only_self_attention(self_attn_type: str) -> None:
    decoder = _decoder(self_attn_type)
    layer = decoder.transformer_layers[0]
    context_keys = torch.arange(16).view(4, 2, 2)
    context_values = context_keys + 100
    layer.context_attn.layer_cache = True, {
        "keys": context_keys,
        "values": context_values,
    }

    if isinstance(layer.self_attn, AverageAttention):
        self_state = torch.arange(16).view(4, 1, 4)
        layer.self_attn.layer_cache = True, {"prev_g": self_state}
    else:
        self_keys = torch.arange(16).view(4, 2, 2)
        self_values = self_keys + 100
        key_pad_mask = torch.tensor([[False, True], [True, False], [False, False], [True, True]])
        layer.self_attn.layer_cache = True, {
            "keys": self_keys,
            "values": self_values,
            "key_pad_mask": key_pad_mask,
        }

    src = decoder.state["src"]
    select_indices = torch.tensor([1, 0, 3, 2])
    decoder.map_state(
        lambda state, dim: state.index_select(dim, select_indices),
        map_src=False,
        map_context=False,
        map_self=True,
    )

    assert decoder.state["src"] is src
    assert layer.context_attn.layer_cache[1]["keys"] is context_keys
    assert layer.context_attn.layer_cache[1]["values"] is context_values
    if isinstance(layer.self_attn, AverageAttention):
        assert torch.equal(
            layer.self_attn.layer_cache[1]["prev_g"],
            self_state.index_select(0, select_indices),
        )
    else:
        assert torch.equal(
            layer.self_attn.layer_cache[1]["keys"],
            self_keys.index_select(0, select_indices),
        )
        assert torch.equal(
            layer.self_attn.layer_cache[1]["values"],
            self_values.index_select(0, select_indices),
        )
        assert torch.equal(
            layer.self_attn.layer_cache[1]["key_pad_mask"],
            key_pad_mask.index_select(0, select_indices),
        )


def test_optimized_mapping_matches_full_mapping_outputs() -> None:
    torch.manual_seed(1)
    full_mapping = _decoder()
    optimized_mapping = _decoder()
    optimized_mapping.load_state_dict(full_mapping.state_dict())
    memory = torch.randn(2, 3, 4).repeat_interleave(2, dim=0)
    memory_mask = torch.tensor([[False, False, True], [False, False, False]]).repeat_interleave(
        2, dim=0
    )
    target = torch.randn(4, 1, 4)
    target_mask = torch.zeros(4, 1, dtype=torch.bool)
    full_mapping(
        target,
        target_mask,
        memory,
        memory_mask,
        step=0,
        return_attn=True,
    )
    optimized_mapping(
        target,
        target_mask,
        memory,
        memory_mask,
        step=0,
        return_attn=True,
    )
    within_source = torch.tensor([1, 0, 3, 2])

    def map_within_source(state, dim):
        return state.index_select(dim, within_source)

    full_mapping.map_state(map_within_source)
    optimized_mapping.map_state(
        map_within_source,
        map_src=False,
        map_context=False,
        map_self=True,
    )

    next_target = torch.randn(4, 1, 4)
    full_output, full_attention = full_mapping(
        next_target,
        target_mask,
        memory.index_select(0, within_source),
        memory_mask.index_select(0, within_source),
        step=1,
        return_attn=True,
    )
    optimized_output, optimized_attention = optimized_mapping(
        next_target,
        target_mask,
        memory,
        memory_mask,
        step=1,
        return_attn=True,
    )
    assert torch.equal(optimized_output, full_output)
    assert torch.equal(optimized_attention["std"], full_attention["std"])

    remaining_source = torch.tensor([2, 3])

    def map_remaining_source(state, dim):
        return state.index_select(dim, remaining_source)

    full_mapping.map_state(map_remaining_source)
    optimized_mapping.map_state(
        map_remaining_source,
        map_src=True,
        map_context=True,
        map_self=True,
    )
    compact_memory = memory.index_select(0, remaining_source)
    compact_mask = memory_mask.index_select(0, remaining_source)
    compact_target = torch.randn(2, 1, 4)
    compact_target_mask = torch.zeros(2, 1, dtype=torch.bool)
    full_output, full_attention = full_mapping(
        compact_target,
        compact_target_mask,
        compact_memory,
        compact_mask,
        step=2,
        return_attn=True,
    )
    optimized_output, optimized_attention = optimized_mapping(
        compact_target,
        compact_target_mask,
        compact_memory,
        compact_mask,
        step=2,
        return_attn=True,
    )
    assert torch.equal(optimized_output, full_output)
    assert torch.equal(optimized_attention["std"], full_attention["std"])


def _beam_search(customised_beam_search: bool) -> BeamSearch:
    beam = BeamSearch(
        pad=0,
        bos=1,
        eos=2,
        unk=3,
        batch_size=2,
        beam_size=2,
        n_best=1,
        max_length=5,
        customised_beam_search=customised_beam_search,
    )
    beam.initialize(torch.zeros(2, 2, 1), torch.tensor([2, 2]))
    return beam


def test_beam_search_requires_initialization_before_updating() -> None:
    with pytest.raises(AssertionError, match="Beam search must be initialized"):
        BeamSearch().update_finished()


@pytest.mark.parametrize("customised_beam_search", [False, True])
def test_beam_search_records_finished_attention(customised_beam_search: bool) -> None:
    beam = _beam_search(customised_beam_search)
    beam.return_attention = True
    log_probs = torch.full((4, 5), -100.0)
    log_probs[0, beam.eos] = 0.0
    log_probs[2, beam.eos] = 0.0
    attention = torch.arange(8, dtype=torch.float).view(1, 4, 2)

    beam.advance(log_probs, attn=attention)
    beam.update_finished()

    assert torch.equal(beam.hypotheses[0][0][2], attention[:, 0, :])
    assert torch.equal(beam.hypotheses[1][0][2], attention[:, 2, :])


def test_beam_search_reports_source_compaction_without_changing_results() -> None:
    beam = _beam_search(customised_beam_search=False)
    log_probs = torch.full((4, 5), -100.0)
    log_probs[0] = torch.tensor([-10.0, -10.0, 0.0, -1.0, -2.0])
    log_probs[2] = torch.tensor([-10.0, -10.0, -2.0, 0.0, -1.0])

    beam.advance(log_probs, attn=None)
    assert beam.update_finished()

    assert beam.batch_offset.tolist() == [1]
    assert beam.select_indices is not None
    assert beam.select_indices.tolist() == [2, 2]
    assert beam.predictions[0][0].tolist() == [beam.eos]
    assert beam.scores[0][0].item() == 0.0
    assert beam.predictions[1] == []


def test_beam_search_does_not_report_compaction_for_single_finished_beam() -> None:
    beam = _beam_search(customised_beam_search=True)
    log_probs = torch.full((4, 5), -100.0)
    log_probs[0] = torch.tensor([-10.0, -10.0, 0.0, -1.0, -2.0])
    log_probs[2] = torch.tensor([-10.0, -10.0, -2.0, 0.0, -1.0])

    beam.advance(log_probs, attn=None)
    assert not beam.update_finished()

    assert beam.batch_offset.tolist() == [0, 1]
    assert beam.select_indices is not None
    assert beam.select_indices.tolist() == [0, 0, 2, 2]


def test_beam_search_does_not_report_compaction_when_all_batches_finish() -> None:
    beam = _beam_search(customised_beam_search=False)
    log_probs = torch.full((4, 5), -100.0)
    log_probs[0] = torch.tensor([-10.0, -10.0, 0.0, -1.0, -2.0])
    log_probs[2] = torch.tensor([-10.0, -10.0, 0.0, -1.0, -2.0])

    beam.advance(log_probs, attn=None)
    assert not beam.update_finished()
    assert beam.done


class _RecordingDecoder:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, dict[str, bool]]] = []

    def init_state(self, src, enc_out, enc_final_hs) -> None:
        pass

    def map_state(self, fn, only_map_src=False, **kwargs) -> None:
        self.calls.append((only_map_src, kwargs))


@pytest.mark.parametrize("parallel_paths", [1, 2])
def test_translator_skips_single_path_mapping_until_compaction(parallel_paths: int) -> None:
    decoder = _RecordingDecoder()
    translator: Any = object.__new__(Translator)
    translator.model = SimpleNamespace(decoder=decoder)
    translator.tgt_prefix = False
    translator.customised_beam_search = False
    translator._tgt_pad_idx = 0
    translator._run_encoder = lambda batch: (
        batch["src"][0],
        None,
        torch.zeros(2, 2, 1),
        batch["src"][1],
    )
    observed_rows: list[int] = []

    def decode(decoder_input, memory_bank, batch, **kwargs):
        step = len(observed_rows)
        observed_rows.append(decoder_input.size(1))
        log_probs = torch.full((decoder_input.size(1), 5), -100.0)
        log_probs[:, 4] = 0.0
        if step == 1:
            log_probs[:parallel_paths, 2] = 1.0
        elif step == 2:
            log_probs[:, 2] = 1.0
        return log_probs, None

    translator._decode_and_generate = decode
    beam = BeamSearch(
        pad=0,
        bos=1,
        eos=2,
        unk=3,
        batch_size=2,
        beam_size=parallel_paths,
        n_best=1,
        max_length=5,
    )
    batch = {
        "src": (torch.ones(2, 2, 1, dtype=torch.long), torch.tensor([1, 2])),
        "batch_size": 2,
    }
    results = translator._translate_batch_with_strategy(batch, beam)

    expected_calls: list[tuple[bool, dict[str, bool]]] = [(True, {})]
    if parallel_paths > 1:
        expected_calls.append((False, {"map_src": False, "map_context": False, "map_self": True}))
    expected_calls.append((False, {"map_src": True, "map_context": True, "map_self": True}))
    assert decoder.calls == expected_calls
    assert observed_rows == [2 * parallel_paths, 2 * parallel_paths, parallel_paths]
    assert results["predictions"][0][0].tolist() == [4, 2]
    assert results["predictions"][1][0].tolist() == [4, 4, 2]
    assert beam.done


@pytest.fixture
def compaction_case():
    """Run two sources through partial finish, source removal, and continued decoding."""
    torch.manual_seed(7)
    memories = (torch.randn(2, 2, 4), torch.randn(2, 2, 4))
    batch = {
        "src": (torch.tensor([[[10], [20]], [[0], [21]]]), torch.tensor([1, 2])),
        "batch_size": 2,
    }

    def run(decoder, decode, tuple_memory=False):
        translator: Any = object.__new__(Translator)
        translator.model = SimpleNamespace(decoder=decoder)
        translator.tgt_prefix = False
        translator.customised_beam_search = True
        translator._tgt_pad_idx = 0
        translator._tgt_eos_idx = 2
        translator._tgt_vocab_len = 6
        translator._run_encoder = lambda batch: (
            batch["src"][0],
            None,
            memories if tuple_memory else memories[0],
            batch["src"][1],
        )
        beam = BeamSearch(
            pad=0,
            bos=1,
            eos=2,
            unk=3,
            batch_size=2,
            beam_size=2,
            n_best=1,
            max_length=6,
            customised_beam_search=True,
        )
        rows = []

        def scheduled_decode(decoder_input, memory_bank, batch, **kwargs):
            step = kwargs["step"]
            rows.append(decoder_input.size(1))
            logits = decode(decoder_input, memory_bank, **kwargs)
            eos_scores = logits[:, 2].clone()
            logits[:, :4] = -100
            first_source = kwargs["batch_offset"].repeat_interleave(2) == 0
            if step == 0:
                logits[first_source, 2] = logits[first_source, 4:6].max(dim=1).values + 1
            elif step == 1:
                logits[first_source, 4:6] = -100
                logits[first_source, 2] = eos_scores[first_source]
            elif step == 3:
                logits[:, 4:6] = -100
                logits[:, 2] = eos_scores
            return torch.log_softmax(logits, dim=-1), None

        translator._decode_and_generate = scheduled_decode
        results = translator._translate_batch_with_strategy(batch, beam)
        assert beam.done
        assert rows == [4, 4, 2, 2]
        return results

    return run, memories, batch


@torch.inference_mode()
def test_translator_decoder_matches_full_mapping(monkeypatch, compaction_case) -> None:
    run, _, batch = compaction_case
    embedding = torch.nn.Embedding(6, 4)
    generator = torch.nn.Linear(4, 6)
    reference, optimized = _decoder(), _decoder()
    optimized.load_state_dict(reference.state_dict())
    original_map = reference.map_state
    monkeypatch.setattr(
        reference,
        "map_state",
        lambda fn, only_map_src=False, **kwargs: original_map(fn, only_map_src=only_map_src),
    )

    def decode_with(decoder, snapshots):
        def decode(tokens, memory, **kwargs):
            source_ids = kwargs["batch_offset"].repeat_interleave(2)
            assert torch.equal(decoder.state["src"], batch["src"][0].index_select(1, source_ids))
            output, _ = decoder(
                embedding(tokens.squeeze(2).transpose(0, 1)),
                torch.zeros(tokens.size(1), 1, dtype=torch.bool),
                memory,
                kwargs["memory_padding_mask"],
                step=kwargs["step"],
            )
            cache = decoder.transformer_layers[0].context_attn.layer_cache[1]
            snapshots.append((decoder.state["src"], cache["keys"], cache["values"]))
            return generator(output.squeeze(1))

        return decode

    snapshots: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    expected = run(reference, decode_with(reference, []))
    actual = run(optimized, decode_with(optimized, snapshots))
    for key in ("predictions", "scores"):
        for expected_batch, actual_batch in zip(expected[key], actual[key]):
            assert len(expected_batch) == len(actual_batch) == 1
            assert torch.equal(expected_batch[0], actual_batch[0])
    for earlier, later in ((0, 1), (2, 3)):
        assert all(a is b for a, b in zip(snapshots[earlier], snapshots[later]))


@pytest.mark.parametrize("tuple_memory", [False, True])
def test_translator_compacts_source_inputs(compaction_case, tuple_memory: bool) -> None:
    run, memories, batch = compaction_case
    observed_banks = []

    def decode(tokens, memory, **kwargs):
        banks = memory if isinstance(memory, tuple) else (memory,)
        source_ids = kwargs["batch_offset"].repeat_interleave(2)
        for bank, original in zip(banks, memories):
            assert torch.equal(bank, original.transpose(0, 1).index_select(0, source_ids))
        lengths = batch["src"][1].index_select(0, source_ids)
        assert torch.equal(kwargs["memory_lengths"], lengths)
        assert torch.equal(kwargs["memory_padding_mask"], torch.arange(2) >= lengths[:, None])
        observed_banks.append(banks)
        return torch.zeros(tokens.size(1), 6)

    run(_RecordingDecoder(), decode, tuple_memory)
    for earlier, later in ((0, 1), (2, 3)):
        assert all(a is b for a, b in zip(observed_banks[earlier], observed_banks[later]))
