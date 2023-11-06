from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[2])
sys.path.append(path_git)
from Source.ModelArgs import *


class SelfAttention(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Number of head for Key and Value
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # Number of head for Query
        self.n_head_q = args.n_heads

        # Indicates how many times the heads of Key and Values should be repeated to match the heaf of the Queries.
        self.n_rep = self.n_head_q // self.n_kv_heads

        # Indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads
        
        self.w_q = torch.nn.Linear(in_features=args.dim,
                                   out_features=self.head_dim * self.n_head_q,
                                   bias=False)
        
        self.w_k = torch.nn.Linear(in_features=args.dim,
                                   out_features=self.head_dim * self.n_kv_heads,
                                   bias=False)

        self.w_v = torch.nn.Linear(in_features=args.dim,
                                   out_features=self.head_dim * self.n_kv_heads,
                                   bias=False)
        
        self.w_o = torch.nn.Linear(in_features=self.head_dim * self.n_head_q,
                                   out_features=args.dim,
                                   bias=False)
        

        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
