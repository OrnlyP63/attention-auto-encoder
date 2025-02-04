import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAutoEncoder(nn.Module):
  def __init__(self, d:int, n:int, l:int):
    super().__init__()

    self.W_q = nn.Parameter(torch.randn(d, d))
    self.W_k = nn.Parameter(torch.randn(d, d))
    self.W_v = nn.Parameter(torch.randn(d, d))

    self.C = nn.Parameter(torch.randn(d, 1))
    self.W_l = nn.Parameter(torch.randn(n, l))
    self.W_l = nn.Parameter(torch.randn(l, n))
    self.d = d

  def forward(self, X_t):
    Q = self.W_q @ X_t
    K = self.W_v @ X_t
    V = self.W_k @ X_t
    K_transpose = torch.transpose(K)
    Z = F.softmax((Q @ K_transpose) / torch.sqrt(self.d))

    A = Z @ V