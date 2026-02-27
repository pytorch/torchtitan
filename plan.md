Option 1: Custom Attributes on Parameters (Simplest)

class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
        quant_config: QuantConfig | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        
        # Attach metadata to parameters
        self.w1._quant_layer_name = "w1"
        self.w2._quant_layer_name = "w2"
        self.w3._quant_layer_name = "w3"
        
        self.use_grouped_mm = use_grouped_mm
        self.quant_config = quant_config

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        # Extract weights and handle DTensor conversion
        if isinstance(self.w1, DTensor):
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.use_grouped_mm:
            # Pass the original parameters (with metadata) instead of converted tensors
            return _run_experts_grouped_mm(
                self.w1, self.w2, self.w3, x, num_tokens_per_expert, self.quant_config
            )
        else:
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)
Then update _run_experts_grouped_mm:


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    quant_config: QuantConfig | None = None,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    
    # Extract actual tensors if DTensor, but keep metadata reference
    def get_tensor_and_name(param):
        if isinstance(param, DTensor):
            tensor = param.to_local()
        else:
            tensor = param
        # Get the layer name from the parameter metadata
        layer_name = getattr(param, '_quant_layer_name', None)
        return tensor, layer_name
    
    def grouped_mm(A: torch.Tensor, B_param: torch.Tensor) -> torch.Tensor:
        B, layer_name = get_tensor_and_name(B_param)
        B_t = B.transpose(-2, -1)
        
        if quant_config is not None and layer_name is not None:
            return quant_config.apply_grouped_mm(A, B_t, offsets, layer_name)
        else:
            return torch._grouped_mm(A.bfloat16(), B_t.bfloat16(), offs=offsets)
    
    h = F.silu(grouped_mm(x, w1))
    h = h * grouped_mm(x, w3)
    out = grouped_mm(h, w2).type_as(x)
    
    return out