from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ModelArgs import *
from Source.model.FeedForwardSwiGLU import *
from Source.model.RMSNorm import *
from Source.model.EncoderBlock import *
from Source.model.InputEmbedding import *

class Transformer(torch.nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        
        assert args.vocab_size != -1, 'vocabsize must be set'

        self.args = args
        self.vocab_size = self.args.vocab_size
        self.n_layers   = self.args.n_layers

        # init input embedding
        self.input_embedding = InputEmbedding(vocab_size=self.vocab_size,
                                                embedding_dim=self.args.dim)

        # init encoder layers : input: x -> x1 = x + Attention[RMSNorm(x)] -> x2 = x1 + FFW[RMSNorm(x1)]
        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args=self.args))


        # init final RMSNorm
        self.final_rmsnorm = RMSNorm(features_dimenstion=self.args.dim,
                                        epsilon=self.args.epsilon,
                                        dropout=self.args.dropout)

        # init final Linear layer
        self.final_linear = torch.nn.Linear(in_features=self.args.dim,
                                            out_features=self.vocab_size,
                                            bias=False)

        self.freqs_complex = precompute_theta_pos_frquencies(head_dimension=self.args.dim // self.args.n_heads,
                                                             seq_len=self.args.max_seq_len)

    def forward(self, x, start_pos):
        # (B, Seq_Len)
        batch_size, seq_len = x.shape
        assert seq_len == 1, "Only one token at a time can be processed"
        # x: [B, 1]
        x = self.input_embedding(x)
        # x: [B, 1, dim]

        # Compute the frequencies for the positional encoding
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len, :]

        for layer in self.layers:
            x = layer(x, start_pos, freqs_complex)
        # x: [B, 1, dim]
        x = self.final_rmsnorm(x)
        # x: [B, 1, dim]
        x = self.final_linear(x)
        # x: [B, 1, vocab_size]
        return x
    

if __name__ == '__main__':
    model_args = ModelArgs()
    model = Transformer(args=model_args).to('cuda')
    print('here')
    print(f'model size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:02.3f}M parameters')
    