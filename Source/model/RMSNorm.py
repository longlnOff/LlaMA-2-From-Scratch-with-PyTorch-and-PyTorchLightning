from pathlib import Path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
path_git = str(Path(current).resolve().parents[2])
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
    a = torch.rand([2, 4, 10])
    rms_test = RMSNorm(10)

    m = rms_test(a)
    n = rms_test.forward1(a)

   
    eq = torch.isclose(input=m, other=n, rtol=13-5, atol=1e-6)
    print(eq)
