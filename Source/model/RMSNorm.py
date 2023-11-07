from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[1])
sys.path.append(path_git)
from Source.ModelArgs import *


class RMSNorm(torch.nn.Module):
    def __init__(self, features_dimenstion: int, epsilon: float = 1e-6, dropout: float = 0.1):
        super().__init__()
        self.features_dimensions = features_dimenstion
        self.epsilon             = epsilon
        self.dropout             = torch.nn.Dropout(dropout)
        self.scale_parameter     = torch.nn.Parameter(torch.ones(self.features_dimensions))


    # Reference
    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward1(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.scale_parameter * self._norm(x.float()).type_as(x)


    # My implementation
    def forward(self, x):
        # x shape = [B, S, F]
        # calculate in F dimension
        square_x = torch.square(x)
        rms_x = torch.mean(input=square_x,
                           dim=-1,
                           keepdim=True)
        x = self.dropout(x * self.scale_parameter / torch.sqrt(rms_x) + self.epsilon)
        # x = x * self.scale_parameter / torch.sqrt(rms_x) + self.epsilon

        return x
    

if __name__ == '__main__':
    model = RMSNorm(features_dimenstion=512,
                           epsilon=1e-5,
                           dropout=0.1)
    print('here')
    print(f'model size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:02.3f}M parameters')
    