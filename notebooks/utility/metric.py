import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def equal_weight_strategy(returns: np.array):
  return np.cumsum(np.mean(returns, axis=1), axis=0)

def anomaly_rebalance_strategy(returns: np.array, anomaly: np.array):
  anomaly = ~anomaly
  portfolio_weights = []
  weight = np.ones(returns.shape[1]) / returns.shape[1]

  for t in range(returns.shape[0]):
    anomaly_t = anomaly[t]
    if np.any(anomaly_t):
      weight = anomaly_t * (1 / anomaly_t.sum())
      portfolio_weights.append(weight)
    else:
      portfolio_weights.append(weight)
  portfolio_weights_array = np.array(portfolio_weights)

  return np.cumsum(np.sum(portfolio_weights_array * returns, axis=1), axis=0)



def plot_profit(returns: np.array):
  n = returns.shape[0]
  sns.lineplot(x=np.arange(n), y=returns, label="Time Series")

  plt.title("Time Series with Anomalies")
  plt.legend()
