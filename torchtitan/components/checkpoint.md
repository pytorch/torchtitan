This document tracks and describes the essential checkpointing features still to be added to TorchTitan.

- [ ] **Full `state_dict` saving**  
  - Support exporting the complete (unsharded) model `state_dict`; many existing formats only handle full `state_dict`.
  - https://github.com/pytorch/torchtitan/pull/1219 is WIP to support this.
  - Need removing FP8 tensor subclass from the `state_dict`.

- [ ] **Model `state_dict` mapping**  
  - Provide an interface for users/developers to plug in custom converters between TorchTitan’s `state_dict`/model definitions and other model definitions (e.g., Hugging Face models).

- [ ] **Hugging Face format saving**  
  - Depends on full `state_dict` export  
  - Optionally leverages the `model state_dict mapping` interface for users who require conversion
  - Uses the Hugging Face API for saving

- [ ] **Hugging Face format loading**  
  - Depends on the `model state_dict interface` as most use cases require conversion from other model definitions  
  - DCP already supports HF loading but needs tighter API integration and performance tuning (collaboration with DCP)

- [ ] **Enhanced checkpoint debugging & comparison tools**  
  - Provide APIs (e.g., per-tensor checksums or diff reports) to pinpoint mismatches in model state, optimizer state, etc.  
  - Streamline root-cause analysis when loaded checkpoints lead to unexpected accuracy changes

- [ ] **Complete unit tests**
  - Checkpointer has a lot of logic and branches. We can verify Checkpointer through Mock without using GPUs.

- [ ] **Decouple `state_dict` staging from checkpointing/DCP calls**  
  - Allow staging of the `state_dict` to CPU (or other targets) independently of DCP 
  - Enables downstream workflows (e.g., RL trainers or parameter servers) to consume staged state without invoking DCP
