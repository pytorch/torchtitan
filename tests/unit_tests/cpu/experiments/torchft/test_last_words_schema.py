# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Required training state is validated before restoring live objects."""

import copy
import json
import pickle

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torchtitan.last_words.job import retention_plan
from torchtitan.last_words.rng import RNG_STATE, RNGCheckpointState
from torchtitan.last_words.schema import state_schema, validate_state_schema
from torchtitan.last_words.validation import recovery_preflight, validate_storage_ranges


def model_payload(step=1):
    # Flat parameter keys match CheckpointManager's full-model DCP layout.
    return {
        "weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "routing_bias": torch.zeros(2),
        "optimizer": {"state": {"weight": {"exp_avg": torch.zeros(2, 2)}}, "param_groups": [{"lr": 0.01}]},
        "lr_scheduler": {"last_epoch": step},
        "ema": {"weight": torch.ones(2, 2)},
        "train_state": {"step": step, "last_words_num_updates": step},
    }


def cursor_payload(replica, step=1):
    rng = RNGCheckpointState(torch.device("cpu"), 0)
    rng.capture()
    return {RNG_STATE: rng, "dataloader": {f"dp_rank_{replica}": b"cursor", "position": step}}


def write_candidate(root, step=1):
    model = model_payload(step)
    dcp.save(model, checkpoint_id=root / "model")
    manifest = dict(
        state=model["train_state"],
        signature="schema-test",
        cursors={},
        rng_schema=1,
        rng_shards=1,
        recovery_schema=state_schema(model, kind="model"),
    )
    for replica in range(2):
        payload = cursor_payload(replica, step)
        path = root / f"cursor-{replica}"
        dcp.save(payload, checkpoint_id=path)
        marker = dict(
            state=model["train_state"],
            signature="schema-test",
            rng_schema=1,
            rng_shards=1,
            recovery_schema=state_schema(payload, kind="cursor"),
        )
        (path / "last_words_cursor.json").write_text(json.dumps(marker))
        manifest["cursors"][str(replica)] = dict(path=path.name, state=model["train_state"])
    return model, manifest


def test_complete_candidate_matches_independent_live_schema(tmp_path):
    model, manifest = write_candidate(tmp_path)
    before = copy.deepcopy(model)
    recovery_preflight(
        tmp_path,
        manifest,
        2,
        expected_model=state_schema(model, kind="model"),
        expected_cursors={"cursor-1": state_schema(cursor_payload(1), kind="cursor")},
    )
    assert torch.equal(model["weight"], before["weight"])
    assert model["train_state"] == before["train_state"]


@pytest.mark.parametrize("missing", ["weight", "routing_bias", "optimizer", "lr_scheduler", "ema", "train_state"])
def test_each_missing_required_state_is_rejected_without_restore(tmp_path, missing):
    model, manifest = write_candidate(tmp_path)
    expected = state_schema(model, kind="model")
    incomplete = {key: value for key, value in model.items() if key != missing}
    target = tmp_path / "incomplete"
    dcp.save(incomplete, checkpoint_id=target)
    with pytest.raises(ValueError, match="required"):
        validate_state_schema(target, manifest["recovery_schema"], kind="model", expected=expected)
    assert torch.equal(model["weight"], torch.arange(4, dtype=torch.float32).reshape(2, 2))


def test_candidate_declaration_cannot_remove_live_model_buffer(tmp_path):
    model, _ = write_candidate(tmp_path)
    expected = state_schema(model, kind="model")
    omitted = {key: value for key, value in model.items() if key != "routing_bias"}
    declared = state_schema(omitted, kind="model")
    dcp.save(omitted, checkpoint_id=tmp_path / "omitted")
    with pytest.raises(ValueError, match="routing_bias"):
        validate_state_schema(tmp_path / "omitted", declared, kind="model", expected=expected)


def test_rng_only_cursor_is_not_a_complete_dataloader_checkpoint(tmp_path):
    _, manifest = write_candidate(tmp_path)
    path = tmp_path / "rng-only"
    payload = cursor_payload(1)
    del payload["dataloader"]
    dcp.save(payload, checkpoint_id=path)
    (path / "last_words_cursor.json").write_text(
        json.dumps(
            dict(
                state=manifest["state"],
                signature="schema-test",
                rng_schema=1,
                rng_shards=1,
                recovery_schema=state_schema(payload, kind="cursor"),
            )
        )
    )
    manifest["cursors"]["1"]["path"] = path.name
    with pytest.raises(ValueError, match="dataloader"):
        recovery_preflight(tmp_path, manifest, 2)


def test_wrong_tensor_dtype_is_rejected(tmp_path):
    model = model_payload()
    expected = state_schema(model, kind="model")
    model["weight"] = model["weight"].double()
    dcp.save(model, checkpoint_id=tmp_path / "wrong-dtype")
    with pytest.raises(ValueError, match="required tensor"):
        validate_state_schema(tmp_path / "wrong-dtype", expected, kind="model")


def test_empty_optimizer_state_and_empty_model_buffer_are_valid(tmp_path):
    model = model_payload(step=0)
    model["optimizer"]["state"] = {}
    model["empty_buffer"] = torch.empty(0)
    schema = state_schema(model, kind="model")
    path = tmp_path / "initial"
    dcp.save(model, checkpoint_id=path)
    validate_storage_ranges(path)
    validate_state_schema(path, schema, kind="model", expected=schema)


def test_malformed_field_description_is_rejected_before_reading_storage(tmp_path):
    schema = state_schema(model_payload(), kind="model")
    schema["fields"]["weight"] = None
    with pytest.raises(ValueError, match="field schema"):
        validate_state_schema(tmp_path / "not-created", schema, kind="model")


@pytest.mark.parametrize("num_shards", [0, 2])
def test_wrong_rng_shard_count_is_rejected_before_restore(tmp_path, num_shards):
    model, manifest = write_candidate(tmp_path)
    manifest["rng_shards"] = num_shards
    with pytest.raises(ValueError, match="RNG|shard"):
        recovery_preflight(tmp_path, manifest, 2)
    assert torch.equal(model["weight"], torch.arange(4, dtype=torch.float32).reshape(2, 2))


def test_missing_tensor_chunk_is_rejected_before_loading(tmp_path):
    path = tmp_path / "missing-chunk"
    dcp.save(model_payload(), checkpoint_id=path)
    metadata_path = path / ".metadata"
    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)
    metadata.state_dict_metadata["weight"].chunks.clear()
    with metadata_path.open("wb") as f:
        pickle.dump(metadata, f)
    with pytest.raises(ValueError, match="chunks are incomplete"):
        validate_storage_ranges(path)


def test_scalar_only_candidate_is_rejected(tmp_path):
    state = {"step": 1, "last_words_num_updates": 1}
    dcp.save({"train_state": state}, checkpoint_id=tmp_path / "model")
    with pytest.raises(ValueError, match="required-state schema"):
        recovery_preflight(tmp_path, dict(state=state, cursors={}, signature="s"), 2)


def test_retention_validates_periodic_rng_without_constructing_manager(tmp_path):
    for step in range(1, 4):
        payload = model_payload(step)
        path = tmp_path / f"step-{step}"
        dcp.save(payload, checkpoint_id=path)
        binding = dict(
            state=payload["train_state"],
            signature="retention",
            run_id="r",
            anchors={},
            rng_schema=1,
            rng_shards=1,
            recovery_schema=state_schema(payload, kind="model"),
        )
        for replica in range(2):
            cursor = tmp_path / f"ft-replicat-{replica}/r/step-{step}"
            data = cursor_payload(replica, step)
            dcp.save(data, checkpoint_id=cursor)
            (cursor / "last_words_cursor.json").write_text(
                json.dumps(
                    dict(
                        state=payload["train_state"],
                        signature="retention",
                        rng_schema=1,
                        rng_shards=1,
                        recovery_schema=state_schema(data, kind="cursor"),
                    )
                )
            )
            binding["anchors"][str(replica)] = cursor.relative_to(tmp_path).as_posix()
        (path / "last_words_binding.json").write_text(json.dumps(binding))
    plan = retention_plan(tmp_path, keep=2, active_seconds=0)
    assert plan["delete"] == [str(tmp_path / "step-1")]
    assert not plan["notes"]
    assert all((tmp_path / f"step-{step}/.metadata").exists() for step in range(1, 4))


def test_retention_includes_elastic_event_candidates(tmp_path):
    for step in range(1, 4):
        path = tmp_path / f"last_words/r/events/event-{step}/replica-0"
        _, manifest = write_candidate(path, step)
        (path / "manifest.json").write_text(json.dumps(manifest))
    plan = retention_plan(tmp_path, keep=2, active_seconds=0)
    assert plan["delete"] == [str(tmp_path / "last_words/r/events/event-1/replica-0")]
    assert not plan["notes"]
