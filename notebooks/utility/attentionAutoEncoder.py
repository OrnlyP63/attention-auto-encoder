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

    K_transpose = K.transpose(-2, -1)
    Z = F.softmax((Q @ K_transpose) / torch.sqrt(self.d), dim=-1)

    A = Z @ V
    C_transpose = self.C.transpose(0, 1)
    S =  C_transpose @ A
    L = F.relu(S @ self.W_l)
    R_hat = L @ self.Beta 

    return R_hat
    
def concate_data(df_dict: Dict[str, pd.DataFrame], 
                #  test_size:float = 0.2
                day_split=None
                 ):
  mask = df_dict['btc'].index
  
  if day_split is None:
    test_size = 0.2
    idx_split = int(len(mask) * test_size)
    traing_idx = mask[:idx_split]
    test_idx = mask[idx_split:]
  else:
    traing_idx = mask <= day_split
    test_idx = mask > day_split

  returns = []
  nvt = []
  mvrv = []

  test_returns = []
  test_nvt = []
  test_mvrv = []

  for tick in df_dict:
    returns.append(df_dict[tick].loc[traing_idx, 'returns'].values)
    nvt.append(df_dict[tick].loc[traing_idx, 'log_nvt'].values)
    mvrv.append(df_dict[tick].loc[traing_idx, 'log_mvrv'].values)

    test_returns.append(df_dict[tick].loc[test_idx, 'returns'].values)
    test_nvt.append(df_dict[tick].loc[test_idx, 'log_nvt'].values)
    test_mvrv.append(df_dict[tick].loc[test_idx, 'log_mvrv'].values)

  returns = np.vstack(returns)
  nvt = np.vstack(nvt)
  mvrv = np.vstack(mvrv)
  res = np.stack([returns, nvt, mvrv], axis=0)

  test_returns = np.vstack(test_returns)
  test_nvt = np.vstack(test_nvt)
  test_mvrv = np.vstack(test_mvrv)
  test_res = np.stack([test_returns, test_nvt, test_mvrv], axis=0)

  return torch.tensor(res, dtype=torch.float), torch.tensor(test_res, dtype=torch.float)

class AttentionDataset(Dataset):
  def __init__(self, data):
      self.data = data

  def __len__(self):
      return self.data.shape[-1]  # Total number of samples

  def __getitem__(self, idx):
      return self.data[:, :, idx]  # Get item at index `idx`
  
def get_return_from_batch(X_t:torch.Tensor):
   m, n = X_t.shape[0], X_t.shape[-1]
   return X_t[:, 0, :].view(m, 1, n)