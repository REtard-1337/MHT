import torch
import torch.nn as nn

class ContextualRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(ContextualRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden_state = torch.zeros((1, embeddings.shape[0], self.rnn.hidden_size), device=embeddings.device)
        rnn_out, _ = self.rnn(embeddings, hidden_state)
        return rnn_out
