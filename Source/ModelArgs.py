from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[0])
sys.path.append(path_git)
print(path_git)
from dataclasses import dataclass
from typing import Optional
import torch
import einops
import math



@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32   # Number of head for queries
    n_kv_heads: Optional[int] = None # Number of head for the KV cache
    vocab_size: int = -1    # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None
