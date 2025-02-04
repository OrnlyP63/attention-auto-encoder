import torch
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

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
    
def concate_data(df_dict: Dict[str, pd.DataFrame]):
  mask = df_dict['btc'].index
  returns = []
  nvt = []
  mvrv = []
  for tick in df_dict:
    returns.append(df_dict[tick].loc[mask, 'returns'].values)
    nvt.append(df_dict[tick].loc[mask, 'log_nvt'].values)
    mvrv.append(df_dict[tick].loc[mask, 'log_mvrv'].values)

  returns = np.vstack(returns)
  nvt = np.vstack(nvt)
  mvrv = np.vstack(mvrv)
  res = np.stack([returns, nvt, mvrv], axis=0)

  return torch.tensor(res, dtype=torch.float)

class AttentionDataset(Dataset):
  def __init__(self, data):
      self.data = data

  def __len__(self):
      return self.data.shape[-1]  # Total number of samples

  def __getitem__(self, idx):
      return self.data[:, :, idx]  # Get item at index `idx`