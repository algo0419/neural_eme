"""PyTorch models for DReME inverse-design experiments."""

from __future__ import annotations

def require_torch():
    """Import torch with a clear dependency error for optional ML utilities."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for forward surrogate training. "
            "Install it with: pip install torch"
        ) from exc
    return torch


torch = require_torch()
nn = torch.nn


class ForwardMLP(nn.Module):
    """Fully connected surrogate mapping normalized geometry vectors to responses."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        layers = []
        last_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(_activation_layer(activation))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _activation_layer(name: str) -> nn.Module:
    normalized = name.strip().lower()
    options = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if normalized not in options:
        allowed = ", ".join(sorted(options))
        raise ValueError(f"Unsupported activation '{name}'. Expected one of: {allowed}.")
    return options[normalized]()
