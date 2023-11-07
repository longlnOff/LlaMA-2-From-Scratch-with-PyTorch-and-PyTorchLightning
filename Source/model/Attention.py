from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ModelArgs import *
from Source.model.RoPE import *


class SelfAttention(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Number of head for Key and Value
        self.n_heads_kv = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # Number of head for Query
        self.n_head_q = args.n_heads

        # Indicates how many times the heads of Key and Values should be repeated to match the heaf of the Queries.
        self.n_rep = self.n_head_q // self.n_heads_kv

        # Indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads
        
        self.w_q = torch.nn.Linear(in_features=args.dim,
                                   out_features=self.head_dim * self.n_head_q,
                                   bias=False)
        
        self.w_k = torch.nn.Linear(in_features=args.dim,
                                   out_features=self.head_dim * self.n_heads_kv,
                                   bias=False)

        self.w_v = torch.nn.Linear(in_features=args.dim,
                                   out_features=self.head_dim * self.n_heads_kv,
                                   bias=False)
        
        self.w_o = torch.nn.Linear(in_features=self.head_dim * self.n_head_q,
                                   out_features=args.dim,
                                   bias=False)
        

        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim)

    def forward(self, x, start_pos, freqs_complex):
        _, seq_len, _ = x.shape    #[B, 1, dim] seq_len always equals one

        # [B, 1, dim] -> [B, 1, head_dim * n_head_q]
        xq = self.w_q(x)
        # [B, 1, dim] -> [B, 1,head_dim *  n_heads_kv]
        xk = self.w_k(x)
        # [B, 1, dim] -> [B, 1,head_dim *  n_heads_kv]
        xv = self.w_v(x)

        # Convert to multi-head
        # [B, 1, head_dim * n_head_q]   -> [B, 1, n_head_q, head_dim]
        xq = einops.rearrange(xq, 'b s (h d) -> b s h d', h=self.n_head_q, d=self.head_dim)
        # [B, 1, head_dim * n_heads_kv] -> [B, 1, n_heads_kv, head_dim]
        xk = einops.rearrange(xk, 'b s (h d) -> b s h d', h=self.n_heads_kv, d=self.head_dim)
        # [B, 1, head_dim * n_heads_kv] -> [B, 1, n_heads_kv, head_dim]
        xv = einops.rearrange(xv, 'b s (h d) -> b s h d', h=self.n_heads_kv, d=self.head_dim)

        # Apply rotary positional embedding for q and k, doesn't change the shape of the tensor
        xq = apply_rotary_embedding(x=xq,
                                    freqs_complex=freqs_complex)
        
        xk = apply_rotary_embedding(x=xk,
                                    freqs_complex=freqs_complex)
        
        # cache k and v
        self.cache_k[:, start_pos: start_pos + seq_len, :, :] = xk
        self.cache_v[:, start_pos: start_pos + seq_len, :, :] = xv

        # get all keys and values from the cache to current position
        # shape = [B, seq_len_kv, n_heads_kv, head_dim]
        keys = self.cache_k[:, 0:(start_pos + seq_len), :, :]
        # shape = [B, seq_len_kv, n_heads_kv, head_dim]
        values = self.cache_v[:, 0:(start_pos + seq_len), :, :]

        # repeat keys and values to match the number of heads of queries
        # shape = [B, seq_len_kv, n_head_kv, head_dim]
        keys = einops.repeat(keys, 'b s h d -> b s (h r) d', r=self.n_rep)
        # shape = [B, seq_len_kv, n_head_q, head_dim]
        values = einops.repeat(values, 'b s h d -> b s (h r) d', r=self.n_rep)
        # shape = [B, seq_len_kv, n_head_q, head_dim]

        # reshape seq_len and n_heads to be multiplied
        # shape = [B, seq_len_q, n_head_q, head_dim]
        xq = xq.transpose(1, 2)
        # shape = [B, seq_len_kv, n_head_q, head_dim]
        keys = keys.transpose(1, 2)
        # shape = [B, seq_len_kv, n_head_q, head_dim]
        values = values.transpose(1, 2)

        # calculate attention
        # shape = [B, n_head_q, seq_len_q, head_dim] @ [B, n_head_q, head_dim, seq_len_kv] --> [B, n_head_q, seq_len_q, seq_len_kv]
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = torch.softmax(scores, dim=-1)

        # [B, n_head_q, seq_len_q, seq_len_kv] @ [B, n_head_q, seq_len_kv, head_dim] --> [B, n_head_q, seq_len_q, head_dim]
        out    = torch.matmul(scores, values)

        # concatenate all heads
        # shape = [B, n_head_q, seq_len_q, head_dim] -> [B, seq_len_q, n_head_q * head_dim]
        out = einops.rearrange(out.contiguous(), 'b h s d -> b s (h d)')

        # [B, seq_len_q, n_head_q * head_dim] -> [B, seq_len_q, dim]
        return self.w_o(out)
    
if __name__ == '__main__':
    model = SelfAttention(args=ModelArgs())
    print('here')
    print(f'model size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:02.3f}M parameters')
    