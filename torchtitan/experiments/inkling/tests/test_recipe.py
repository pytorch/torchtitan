# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from torchtitan.config import ConfigLoader
from torchtitan.experiments.inkling import trainer as inkling_trainer
from torchtitan.trainer import Trainer
from torchtitan_recipes.models.inkling import inkling_2_mini_lora


_SOURCE_FILES = (
    "model.safetensors",
    "tokenizer.json",
    "mtp.safetensors",
    "hf_quant_config.json",
)


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    for name in _SOURCE_FILES:
        (source / name).write_bytes(b"aaa")
    corpus = tmp_path / "train.txt"
    corpus.write_text("one two three\n", encoding="utf-8")
    return source, corpus, tmp_path / "run"


def _launch_config(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    corpus: Path,
    run: Path,
    *,
    steps: int = 10,
    seq_len: int = 8,
):
    def spark():
        return inkling_2_mini_lora(
            checkpoint=source, text=corpus, seq_len=seq_len, steps=steps
        )

    monkeypatch.setattr(sys.modules[__name__], "spark", spark, raising=False)
    return ConfigLoader().load(
        ["--module", __name__, "--config", "spark", "--output-dir", str(run)]
    )


@pytest.mark.parametrize("changed", _SOURCE_FILES)
def test_resume_requires_original_checkpoint_contents(monkeypatch, inputs, changed):
    monkeypatch.setattr(Trainer, "__init__", lambda self, config: None)
    source, corpus, run = inputs
    _launch_config(monkeypatch, *inputs).build()
    (run / "checkpoint" / "step-10").mkdir(parents=True)

    relocated = source.with_name("relocated")
    shutil.copytree(source, relocated)
    _launch_config(monkeypatch, relocated, corpus, run, steps=20).build()
    (relocated / changed).write_bytes(b"bbb")
    with pytest.raises(ValueError, match="different checkpoint contents"):
        _launch_config(monkeypatch, relocated, corpus, run, steps=20).build()
    _launch_config(monkeypatch, *inputs, steps=20).build()


@pytest.mark.parametrize("changed", ["corpus contents", "seq_len"])
def test_resume_binds_the_data_cursor_to_corpus_and_context(
    monkeypatch, inputs, changed
):
    monkeypatch.setattr(Trainer, "__init__", lambda self, config: None)
    source, corpus, run = inputs
    _launch_config(monkeypatch, *inputs).build()
    (run / "checkpoint" / "step-10").mkdir(parents=True)

    relocated = corpus.with_name("relocated.txt")
    shutil.copyfile(corpus, relocated)
    _launch_config(monkeypatch, source, relocated, run, steps=20).build()
    seq_len = 8
    if changed == "corpus contents":
        relocated.write_text("one two seven\n", encoding="utf-8")
    else:
        seq_len = 16
    with pytest.raises(ValueError, match=f"different {changed}"):
        _launch_config(
            monkeypatch, source, relocated, run, seq_len=seq_len, steps=20
        ).build()


@pytest.mark.parametrize(
    ("literal", "alternate"),
    [("train[1].txt", "train1.txt"), ("train.txt::alt.txt", "train.txt")],
)
def test_rejects_ambiguous_corpus_path(monkeypatch, inputs, literal, alternate):
    source, corpus, run = inputs
    corpus = corpus.rename(corpus.with_name(literal))
    corpus.with_name(alternate).write_text("wrong document\n", encoding="utf-8")

    def init(trainer, config):
        dataset = config.dataloader.dataset.dataset.source.build()
        pytest.fail(f"Unexpectedly loaded {dataset[0]['text']!r}")

    monkeypatch.setattr(Trainer, "__init__", init)
    with pytest.raises(ValueError, match="glob|chained"):
        _launch_config(monkeypatch, source, corpus, run).build()
    assert not (run / "run-manifest.json").exists()


@pytest.mark.parametrize("change", ["shuffle", "packing", "filter"])
def test_resume_rejects_changed_document_order_or_packing(monkeypatch, inputs, change):
    monkeypatch.setattr(Trainer, "__init__", lambda self, config: None)
    _launch_config(monkeypatch, *inputs).build()
    config = _launch_config(monkeypatch, *inputs, steps=20)
    if change == "shuffle":
        config.dataloader.shuffle = True
    elif change == "filter":
        packing = config.dataloader.dataset
        config.dataloader.dataset = replace(
            packing, dataset=replace(packing.dataset, pre_filters=(lambda row: False,))
        )
    else:
        config.dataloader.max_num_documents = 2
    with pytest.raises(ValueError, match="document"):
        config.build()


def test_refuses_to_adopt_unverified_native_checkpoint(monkeypatch, inputs):
    monkeypatch.setattr(Trainer, "__init__", lambda self, config: None)
    _, _, run = inputs
    (run / "checkpoint" / "step-1").mkdir(parents=True)
    with pytest.raises(ValueError, match="missing run manifest"):
        _launch_config(monkeypatch, *inputs).build()
    assert not (run / "run-manifest.json").exists()


@pytest.mark.parametrize("change_source", [False, True], ids=["unchanged", "changed"])
def test_native_training_exports_only_from_verified_source(
    monkeypatch, inputs, change_source
):
    source, _, run = inputs

    def init(trainer, config):
        model = torch.nn.Module()
        model.attn = torch.nn.Module()
        model.attn.lora_b = torch.nn.Linear(1, 1, bias=False)
        trainer.config = config
        trainer.engine = SimpleNamespace(num_completed_steps=0, model_parts=[model])

    def train(trainer):
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        trainer.engine.num_completed_steps = trainer.config.training.steps
        with torch.no_grad():
            trainer.engine.model_parts[0].attn.lora_b.weight.fill_(1)
        if change_source:
            (source / "model.safetensors").write_bytes(b"changed during training")

    monkeypatch.setattr(Trainer, "__init__", init)
    monkeypatch.setattr(Trainer, "train", train)
    monkeypatch.setattr(
        inkling_trainer,
        "export_merged_checkpoint",
        lambda model, src, dest: dest.mkdir(),
    )
    trainer = _launch_config(monkeypatch, *inputs).build()
    export = run / "merged-step-10"
    if change_source:
        with pytest.raises(ValueError, match="different checkpoint contents"):
            trainer.train()
        assert not export.exists()
    else:
        trainer.train()
        assert export.is_dir()
        with pytest.raises(ValueError, match="Merged checkpoint already exists"):
            _launch_config(monkeypatch, *inputs).build()


def test_recipe_tokenizer_preserves_hf_tokens_and_loads_generation_tags(
    monkeypatch, inputs
):
    source, _, _ = inputs
    tokenizer = Tokenizer(WordLevel({"hello": 0, "world": 1, "<EOS>": 2}))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(source / "tokenizer.json"))
    (source / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<EOS>"}), encoding="utf-8"
    )
    (source / "chat_template.jinja").write_text(
        "{% for message in messages %}  "
        "{%- generation -%}{{ message.content }} {{ eos_token }}{%- endgeneration -%}"
        "{% endfor %}",
        encoding="utf-8",
    )
    config = _launch_config(monkeypatch, *inputs)
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    assert tokenizer.encode("hello world", add_bos=True, add_eos=True) == [0, 1, 2]
    messages = [{"role": "assistant", "content": "hello world"}]
    assert tokenizer.apply_chat_template(messages) == "hello world <EOS>"
