from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ModelArgs import *
from Source.model.RMSNorm import *
from Source.model.Attention import *
from Source.model.FeedForwardSwiGLU import *

class EncoderBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads    = args.n_heads
        self.dim        = args.dim
        self.head_dim   = self.dim // self.n_heads


        # RMSNorm before self-attention
        self.rmsnorm_bf_att     = RMSNorm(features_dimenstion=self.dim,
                                      epsilon=args.epsilon,
                                      dropout=args.dropout)
        
        # Self-Attention layer
        self.attention_layer    = SelfAttention(args=args)

        # RMSNorm before FFW
        self.rmsnorm_bf_ffw     = RMSNorm(features_dimenstion=self.dim,
                                      epsilon=args.epsilon,
                                      dropout=args.dropout)
        
        # FFW layer
        self.ffw                = FeedForward(args=args)


    def forward(self, x, start_pos, freqs_complex):
        x1 = self.attention_layer(self.rmsnorm_bf_att(x), start_pos, freqs_complex)
        x1 = x + x1

        x2 = self.ffw(self.rmsnorm_bf_ffw(x1))
        x2 = x1 + x2

        return x2 
    

if __name__ == '__main__':
    model = EncoderBlock(args=ModelArgs())
    print('here')
    print(f'model size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:02.3f}M parameters')
    