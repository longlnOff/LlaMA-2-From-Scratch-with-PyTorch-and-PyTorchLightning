from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[2])
sys.path.append(path_git)
from Source.ModelArgs import *
print(path_git)

class Transformer(torch.nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        
        assert args.vocab_size != -1, 'vocabsize must be set'

        self.args = args
        self.vocab_size = self.args.vocab_size
        self.n_layers   = self.args.n_layers

        # init input embedding

        # init rotary positional embedding
        
        # init encoder layers : input: x -> x1 = x + Attention[RMSNorm(x)] -> x2 = x1 + FFW[RMSNorm(x1)]

        # init final RMSNorm

        # init final Linear layer

    def forward(self, x):
        pass