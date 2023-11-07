from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ModelArgs import *


def precompute_theta_pos_frquencies(head_dimension: int, seq_len: int, theta: float = 10000.0):
    assert head_dimension % 2 == 0, "head_dimension must be divisible by 2"
    
    
    d_divide_2 = torch.arange(0, head_dimension // 2, 1, dtype=torch.float32)
    # shape = [head_dimension / 2]
    # print(d_divide_2)
    theta_matrix = theta ** (-2 * (d_divide_2) / float(head_dimension))
    # shape = [head_dimension / 2]

    m = torch.arange(seq_len)
    # shape = [seq_len]

    # position encode
    pos_enc = torch.outer(m, theta_matrix).float()
    # shape = [seq_len, head_dimension / 2]

    # convert to complex numbers in the polar form c = R * exp(m * theta) where R  = 1:
    pos_enc_complex = torch.polar(torch.ones_like(pos_enc), pos_enc)
    # shape = [seq_len, head_dimension / 2]

    return pos_enc_complex


def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor):
    # x shape = [B, seq_len, H, dimension]
    # freqs_complex shape = [seq_len, dimension / 2]
    
    # convert x to each pair 2 block (2 consecutive values will become a single complex number)
    x_complex = torch.view_as_complex_copy(x.float().reshape(*x.shape[:-1], -1, 2))
    # shape = [B, seq_len, H, dimension / 2]

    # reshape freqs_complex to match x dimension
    freqs_complex = einops.rearrange(freqs_complex, 's d -> 1 s 1 d')
    # shape = [1, seq_len, 1, dimension / 2]

    # multiply each complex number in the x_complex tensor by the corresponding number 
    # in freqs_complex, which results in the rotation of the complex number as shown in the 
    # Figure 1 of the paper.
    x_rotated = x_complex * freqs_complex
    # shape = [B, seq_len, H, dimension / 2]        # in complex form (which contain real and complex segment)
    
    x_out = torch.view_as_real(x_rotated)
    # shape = [B, seq_len, H, dimension]

    x_out = x_out.reshape(*x.shape)
    # shape = [B, seq_len, H, dimension]

    return x_out

if __name__ == '__main__':
    head_dimension = 512 // 4
    seq_len = 1024
    a = precompute_theta_pos_frquencies(head_dimension=head_dimension,
                                        seq_len=seq_len)
    
    x = torch.randn(1, seq_len, 4, head_dimension)

    print(x.shape)
    print(a.shape)

    print(apply_rotary_embedding(x, a).shape)