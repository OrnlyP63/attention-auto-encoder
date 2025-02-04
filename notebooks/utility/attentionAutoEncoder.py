import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAutoEncoder(nn.Module):
  def __init__(self, d:int, n:int, l:int):
    super().__init__()

    self.W_q = nn.Parameter(torch.randn(d, d), requires_grad=True)
    self.W_k = nn.Parameter(torch.randn(d, d), requires_grad=True)
    self.W_v = nn.Parameter(torch.randn(d, d), requires_grad=True)

    self.C = nn.Parameter(torch.randn(d, 1), requires_grad=True)
    self.W_l = nn.Parameter(torch.randn(n, l), requires_grad=True)
    self.Beta = nn.Parameter(torch.randn(l, n), requires_grad=True)
    self.d = nn.Parameter(torch.tensor([d]), requires_grad=False)

  def forward(self, X_t):
    Q = self.W_q @ X_t
    K = self.W_v @ X_t
    V = self.W_k @ X_t

    K_transpose = K.transpose(0, 1)
    Z = F.softmax((Q @ K_transpose) / torch.sqrt(self.d), dim=-1)

    A = Z @ V
    C_transpose = self.C.transpose(0, 1)
    S =  C_transpose @ A
    L = F.relu(S @ self.W_l)
    R_hat = L @ self.Beta 

    return R_hat