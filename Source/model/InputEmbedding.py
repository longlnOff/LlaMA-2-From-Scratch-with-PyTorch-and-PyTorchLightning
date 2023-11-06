from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[2])
sys.path.append(path_git)
from Source.ModelArgs import *


class InputEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, input):
        # input shape = [B, seq_len]
        output = self.embedding(input)
        # output shape = [B, seq_len, embedding_dim]
        return output * math.sqrt(self.embedding_dim)