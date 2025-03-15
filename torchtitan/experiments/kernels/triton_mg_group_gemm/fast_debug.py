import logging

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import the grouped GEMM implementations
try:
    from mg_backward import group_gemm_backward as grouped_gemm_backward
    from mg_forward import group_gemm_forward as grouped_gemm
except ImportError:
    logging.error(
        "Error importing grouped GEMM modules. Make sure the implementation files are in the correct path."
    )
    raise


def test_backward_pass():
    """
    A simple test for the M*G grouped GEMM backward pass with detailed error handling.

    In M*G grouping:
    - M dimension is partitioned into G groups (M_total = sum(M_sizes))
    - N dimension is the same for all groups
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test parameters for DeepSeek-like models
        G = 4  # Number of groups
        M_sizes = [1024, 1024, 2048, 2048]  # Group sizes (will be adjusted)
        M_total = sum(M_sizes)  # Total M dimension
        N = 512  # Output dimension (same for all groups)
        K = 256  # Hidden dimension

        # Deepseek-like configs: ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168))
        # Format: (G, M, K, N)

        # Create group sizes tensor
        m_sizes = torch.tensor(M_sizes, device=device, dtype=torch.int32)

        # Create input and weight tensors
        x = torch.randn(
            M_total, K, dtype=torch.float16, device=device, requires_grad=True
        )
        w = torch.randn(N, K, dtype=torch.float16, device=device, requires_grad=True)

        # Log the setup
        logging.info(f"Test setup - G: {G}, M_total: {M_total}, N: {N}, K: {K}")
        logging.info(f"Group sizes: {m_sizes}")
        logging.info(f"Input x shape: {x.shape}")
        logging.info(f"Weight w shape: {w.shape}")

        # Step 1: Run forward pass
        logging.info("Running forward pass")
        result = grouped_gemm(x, w, m_sizes)
        logging.info(f"Forward result shape: {result.shape}")

        # Create a gradient for backpropagation
        grad_output = torch.randn_like(result)
        logging.info(f"Created gradient with shape: {grad_output.shape}")

        # Step 2: Run backward pass directly
        logging.info("Running backward pass directly")
        grad_x, grad_w = grouped_gemm_backward(grad_output, x, w, m_sizes)

        # Verify gradient shapes
        logging.info(
            f"Gradient shapes - grad_x: {grad_x.shape}, grad_w: {grad_w.shape}"
        )

        # Step 3: Verify gradient computation using PyTorch's autograd
        # First create autograd-enabled tensors
        x_autograd = x.detach().clone().requires_grad_(True)
        w_autograd = w.detach().clone().requires_grad_(True)

        # Create a PyTorch reference implementation to compare against
        logging.info("Running PyTorch reference implementation")

        # Compute reference result
        reference_result = torch.zeros_like(result)
        m_start = 0
        for g in range(G):
            m_size = m_sizes[g].item()
            if m_size > 0:
                m_end = m_start + m_size

                # Extract group input
                x_g = x_autograd[m_start:m_end]

                # Compute group output: y_g = x_g @ w.T
                y_g = torch.matmul(x_g, w_autograd.T)

                # Store result
                reference_result[m_start:m_end] = y_g

                # Update start index
                m_start = m_end

        # Backpropagate using PyTorch
        reference_result.backward(grad_output)

        # Compare gradients
        logging.info("Comparing gradients with PyTorch reference")
        grad_x_error = (grad_x - x_autograd.grad).abs().max().item()
        grad_w_error = (grad_w - w_autograd.grad).abs().max().item()

        logging.info(
            f"Maximum gradient error - grad_x: {grad_x_error}, grad_w: {grad_w_error}"
        )

        # Check if gradients are close using allclose
        rtol = 1e-1  # Relative tolerance for bfloat16
        atol = 1e-1  # Absolute tolerance for bfloat16

        grad_x_close = torch.allclose(grad_x, x_autograd.grad, rtol=rtol, atol=atol)
        if not grad_x_close:
            logging.warning("FAILED: Gradient mismatch detected in grad_x")
        else:
            logging.info(
                "✓ SUCCESS! grad_X matches the PyTorch reference (allclose check passed)"
            )

        grad_w_close = torch.allclose(grad_w, w_autograd.grad, rtol=rtol, atol=atol)
        if not grad_w_close:
            logging.warning("FAILED: Gradient mismatch detected in grad_w")
        else:
            logging.info(
                "✓ SUCCESS! grad_W matches the PyTorch reference (allclose check passed)"
            )

        logging.info(
            f"Gradients allclose check - grad_x: {grad_x_close}, grad_w: {grad_w_close}"
        )

        if grad_x_close and grad_w_close:
            logging.info(
                "✓ SUCCESS: Gradients match the PyTorch reference (allclose check passed)"
            )
        else:
            logging.error("✗ FAILURE: Gradient mismatch detected in allclose check")

        # Additional diagnostics for all cases
        # Import numpy for unravel_index
        import numpy as np

        # Analyze grad_x
        diff_x = (grad_x - x_autograd.grad).abs()
        max_idx_x = diff_x.argmax().item()
        idx_x = np.unravel_index(max_idx_x, grad_x.shape)
        logging.info(
            f"Largest grad_x difference at {idx_x}: "
            f"{grad_x[idx_x].item()} vs {x_autograd.grad[idx_x].item()}"
        )

        # Count zeros in grad_x
        zeros_grad_x = (grad_x == 0).sum().item()
        zeros_autograd_x = (x_autograd.grad == 0).sum().item()
        logging.info(
            f"Zeros in grad_x: {zeros_grad_x}/{grad_x.numel()} ({zeros_grad_x/grad_x.numel()*100:.2f}%)"
        )
        logging.info(
            f"Zeros in x_autograd.grad: {zeros_autograd_x}/{x_autograd.grad.numel()} ({zeros_autograd_x/x_autograd.grad.numel()*100:.2f}%)"
        )

        # Analyze grad_w
        diff_w = (grad_w - w_autograd.grad).abs()
        max_idx_w = diff_w.argmax().item()
        idx_w = np.unravel_index(max_idx_w, grad_w.shape)
        logging.info(
            f"Largest grad_w difference at {idx_w}: "
            f"{grad_w[idx_w].item()} vs {w_autograd.grad[idx_w].item()}"
        )

        # Count zeros in grad_w
        zeros_grad_w = (grad_w == 0).sum().item()
        zeros_autograd_w = (w_autograd.grad == 0).sum().item()
        logging.info(
            f"Zeros in grad_w: {zeros_grad_w}/{grad_w.numel()} ({zeros_grad_w/grad_w.numel()*100:.2f}%)"
        )
        logging.info(
            f"Zeros in w_autograd.grad: {zeros_autograd_w}/{w_autograd.grad.numel()} ({zeros_autograd_w/w_autograd.grad.numel()*100:.2f}%)"
        )

        # Check for NaN values (could indicate numerical issues)
        nan_x = torch.isnan(grad_x).sum().item()
        nan_w = torch.isnan(grad_w).sum().item()
        if nan_x > 0 or nan_w > 0:
            logging.error(f"NaN values detected! grad_x: {nan_x}, grad_w: {nan_w}")

        return grad_x_close and grad_w_close

    except Exception as e:
        logging.error(f"Test failed with error: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return False


def test_multiple_deepseek_configs():
    """
    Test multiple DeepSeek model configurations.
    """
    # DeepSeek configurations: (G, M, K, N)
    configs = [
        (4, 8192, 7168, 4096),  # Config 1
        (4, 8192, 2048, 7168),  # Config 2
        (8, 4096, 7168, 4096),  # Config 3
        (8, 4096, 2048, 7168),  # Config 4
    ]

    results = []

    for config_idx, (G, M, K, N) in enumerate(configs):
        logging.info(f"\n\n===== Testing DeepSeek Config {config_idx+1} =====")
        logging.info(f"G={G}, M={M}, K={K}, N={N}")

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Create even group sizes
            base_size = M // G
            remainder = M % G
            M_sizes = [base_size + (1 if i < remainder else 0) for i in range(G)]
            m_sizes = torch.tensor(M_sizes, device=device, dtype=torch.int32)

            # Create input and weight tensors
            x = torch.randn(
                M, K, dtype=torch.bfloat16, device=device, requires_grad=True
            )
            w = torch.randn(
                N, K, dtype=torch.bfloat16, device=device, requires_grad=True
            )

            logging.info(f"Input x shape: {x.shape}, Weight w shape: {w.shape}")

            # Run forward pass
            result = grouped_gemm(x, w, m_sizes)
            logging.info(f"Forward result shape: {result.shape}")

            # Create gradient for backpropagation
            grad_output = torch.randn_like(result)

            # Run backward pass
            grad_x, grad_w = grouped_gemm_backward(grad_output, x, w, m_sizes)

            # Setup PyTorch reference
            x_ref = x.detach().clone().requires_grad_(True)
            w_ref = w.detach().clone().requires_grad_(True)

            # Compute reference result
            ref_result = torch.zeros_like(result)
            m_start = 0
            for g in range(G):
                m_size = M_sizes[g]
                if m_size > 0:
                    m_end = m_start + m_size
                    x_g = x_ref[m_start:m_end]
                    y_g = torch.matmul(x_g, w_ref.T)
                    ref_result[m_start:m_end] = y_g
                    m_start = m_end

            # Backpropagate
            ref_result.backward(grad_output)

            # Compare
            rtol = 1e-1
            atol = 1e-1
            grad_x_close = torch.allclose(grad_x, x_ref.grad, rtol=rtol, atol=atol)
            grad_w_close = torch.allclose(grad_w, w_ref.grad, rtol=rtol, atol=atol)

            # Log results
            if grad_x_close and grad_w_close:
                logging.info(f"✓ SUCCESS: Config {config_idx+1} passed!")
            else:
                logging.error(f"✗ FAILURE: Config {config_idx+1} failed!")
                if not grad_x_close:
                    logging.error("  grad_x mismatch")
                if not grad_w_close:
                    logging.error("  grad_w mismatch")

            results.append((config_idx + 1, grad_x_close and grad_w_close))

        except Exception as e:
            logging.error(f"Config {config_idx+1} test failed with error: {e}")
            import traceback

            logging.error(traceback.format_exc())
            results.append((config_idx + 1, False))

    # Summary
    logging.info("\n===== Test Results Summary =====")
    for config_idx, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logging.info(f"Config {config_idx}: {status}")

    return all(success for _, success in results)


if __name__ == "__main__":
    logging.info("Running fast debug for M*G grouped GEMM")

    # Import numpy for unravel_index
    import numpy as np

    # Run single test
    logging.info("\n===== Running basic backward pass test =====")
    success_basic = test_backward_pass()
    logging.info(f"Basic test {'succeeded' if success_basic else 'failed'}")

    # Run multiple DeepSeek configs
    logging.info("\n===== Running tests for all DeepSeek configs =====")
    success_configs = test_multiple_deepseek_configs()
    logging.info(
        f"DeepSeek configs tests {'all succeeded' if success_configs else 'had failures'}"
    )

    # Overall result
    overall_success = success_basic and success_configs
    logging.info(
        f"\nOverall test result: {'SUCCESS' if overall_success else 'FAILURE'}"
    )
