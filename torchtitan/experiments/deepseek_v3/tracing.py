# torchrun --nproc-per-node 4 --standalone tracing.py

import torch
import torch.distributed as dist

from model import MoE
from model_config import deepseek_config_registry


def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print("\n")
        print(*args, **kwargs)

def setup_mesh():
    ep_size = dist.get_world_size()
    mesh_shape = (ep_size,)
    mesh = dist.init_device_mesh("cuda", mesh_shape, mesh_dim_names=("ep",))
    return mesh

def setup_model(mesh):
    group_size = mesh["ep"].size()
    rank = mesh["ep"].get_local_rank()

    model_id = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    config = deepseek_config_registry[model_id]
    config.ep_size = group_size

    device = torch.device("cuda", rank % torch.cuda.device_count())
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)

    # Initialize the model
    print0("Initializing MoE model...")
    with mesh, torch.device(device):
        moe = MoE(config)

    print0("Setting up Symmetric Memory ...")
    moe.setup_symm_mem(torch.bfloat16, device)

    return moe

def test_export(moe, mesh):
    seqlen = 256
    batch_size = 1
    config = moe.config

    rank = mesh["ep"].get_local_rank()
    device = torch.device("cuda", rank % torch.cuda.device_count())

    x = torch.randn(
        batch_size, seqlen, config.hidden_size, dtype=torch.bfloat16, device=device
    )
    y = moe(x)
    # print(y.shape)

    # Let's export the model
    print0("Exporting MoE model using torch.export...")

    # Put model in eval mode for export
    moe.eval()

    # Create example input for export
    example_input = (
        torch.randn(
            batch_size, seqlen, config.hidden_size, dtype=torch.bfloat16, device=device
        ),
    )

    # Export using torch.export.export
    exported_model = torch.export.export(moe, example_input)
    print0("Successfully exported the MoE model using torch.export")

    # Save the exported model
    # export_path = "exported_moe_model.pt"
    # torch.export.save(exported_model, export_path)
    # print(f"Exported model saved to: {export_path}")

    # Test the exported model
    print0("Testing exported model...")
    with torch.no_grad():
        original_output = moe(*example_input)
        exported_output = exported_model.module()(*example_input)

        # Check if outputs are close
        if torch.allclose(original_output, exported_output, rtol=1e-3, atol=1e-3):
            print0("✓ Exported model outputs match original model outputs")
        else:
            print0("⚠ Exported model outputs differ from original model")
            print0(
                f"Max difference: {torch.max(torch.abs(original_output - exported_output))}"
            )

    print0("Model export completed!\n")

    if rank == 0:
        exported_model.graph_module.print_readable()


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    mesh = setup_mesh()
    moe = setup_model(mesh)
    test_export(moe, mesh)
    torch.distributed.destroy_process_group()
