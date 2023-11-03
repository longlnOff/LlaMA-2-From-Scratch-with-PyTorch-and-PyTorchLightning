import torch

def precompute_theta_pos_frquencies(head_dim: int, seq_len: int, theta: float = 10000.0):
    assert head_dim % 2 == 0, "head_dim must be divisible by 2"
    
    matrix_index = torch.arange(0, head_dim, 2, dtype=torch.float32)
    theta_matrix = theta ** (-2 * (matrix_index) / float(head_dim))
    