# Changelog

Changelog for development of mlperf flux benchmark.

## 2025-06-18
- Fixed bug where coco data was being saved as jpg after resizing, causing the validation loss to diverge
- Fixed issues where preprocessed data was being generated with dropout
- Fixed mlperf logging step count bug

## 2025-06-05
- Fixed bug to lr scheduler

## 2025-06-03
- QOL improvements to run.sub and eval.sub
- default to fsdp within node, dp outside node (HSDP)
- gather instead of all_gather for eval and infer

## 2025-05-27
- Made run.sub default to full FSDP
- Introduced dummy dataset
- Fixed inference and evaluation scripts

## 2025-05-23
- Updated dockerfile to install torch nightly. This made FSDP now scale as expected, whereas before it was broken.
- Refactored inference.
- Added jobchain script.
- Added configs for training different models.
- Introduced argument to pass dimension of text encoder, for compatibility with different encoders.

## 2025-04-30
- Fixed bug that was setting eval dataset path = train dataset path when this was set in the config.

## First entry 2025-04-29

Despite this being the first entry, a few notable changes have already been made. This will attempt to summarize them.

### Changes required to be upstreamed / updated once available upstream
- Implementation of validation loop / logic (https://github.com/pytorch/torchtitan/issues/1150)
- Fixes regarding dtype compatibility with FSDP (https://github.com/pytorch/torchtitan/issues/1137)
- Fixed confusion bug where seq_len is used to set the hidden_dim of t5 encoder (https://github.com/pytorch/torchtitan/issues/1146)
- Creation of Dockerfile
- Make batch generation in train.py more pythonic (https://github.com/pytorch/torchtitan/issues/1139)
- Batchify inference

### Changes relevant to MLPerf reference
- Integration of laion, coco datasets.
- Modify debug_model to use a smaller version of t5
- Creation of initial eval script to compute FID and CLIP scores from a checkpoint
