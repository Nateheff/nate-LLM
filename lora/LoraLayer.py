import torch
import torch.nn as nn
import torch.nn.functional as F
from LoraConfig import LoraConfig



class LoraLayer(nn.Module):
    def __init__(self, rank:int, head_dim:int, n_embd:int) -> None:
        super().__init__()
        self.rank = rank
        self.head_dim = head_dim
        self.A = nn.Parameter(torch.randn((self.rank, self.head_dim)))
        self.B = nn.Parameter(torch.zeros((self.head_dim, self.rank)))
        self.matrix = self.A @ self.B
        self.final_test = nn.Linear(n_embd, self.head_dim)

    def forward(self, x):

        out = F.linear(x, self.matrix)
        return out

