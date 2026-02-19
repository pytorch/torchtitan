## `ForgeEngine`

The `forge` folder contains a lightweight training engine that serves as a streamlined subset of the `Trainer` class from [torchtitan/train.py](/torchtitan/train.py). This engine provides only the essential constructor method, making it highly flexible for various downstream applications.

The [`ForgeEngine`](engine.py) takes a `ForgeEngine.Config` to
- Initialize an SPMD distributed training environment
- Construct and scale models via n-D parallelisms and meta-device initialization
- Provide necessary training components and utilities

**Primary Use Case**: The engine is designed for building trainers in post-training workflows where multiple specialized components (trainer, generator, replay buffer, parameter server, etc.) work together.

Additionally, the folder provides a model spec registration method [`register_model_spec`](model_spec.py) that allows users to extend beyond the core set of models and training components available in torchtitan, enabling greater flexibility and customization for specific training requirements.

The [example_train.py](./example_train.py) demonstrates how to use `ForgeEngine` for pretraining, achieving the same functionality as [torchtitan/train.py](/torchtitan/train.py) (except for quantization or fault tolerance).
