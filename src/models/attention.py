import torch
import torch.nn as nn


class MultiFacetedAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(MultiFacetedAttention, self).__init__()
        self.thematic_attn = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        self.emotional_attn = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
        self.contextual_attn = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        thematic_out, _ = self.thematic_attn(x, x, x)
        emotional_out, _ = self.emotional_attn(x, x, x)
        contextual_out, _ = self.contextual_attn(x, x, x)

        return thematic_out + emotional_out + contextual_out
