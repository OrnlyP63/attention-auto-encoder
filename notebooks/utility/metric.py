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



def plot_profit(returns_list: list, labels: list):
    plt.figure(figsize=(10, 6))  # Set figure size
    
    for returns, label in zip(returns_list, labels):
        n = returns.shape[0]
        sns.lineplot(x=np.arange(n), y=returns, label=label)
    
    plt.title("Cumulative Strategy Returns Over Time", fontsize=14)
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Cumulative Returns", fontsize=12)
    
    plt.legend()
    plt.grid(True)  # Add grid for better readability
    # plt.show()  # Ensure display of the plot

def anomaly_rebalance_strategy2(returns: np.array, anomaly: np.array):
  anomaly = ~anomaly[:-1, :]
  portfolio_weights = [np.ones(anomaly.shape[1]) / anomaly.shape[1]]
  weight = np.ones(returns.shape[1]) / returns.shape[1]

  for t in range(anomaly.shape[0]):
    anomaly_t = anomaly[t]
    if np.any(anomaly_t):
      weight = anomaly_t * (1 / anomaly_t.sum())
      portfolio_weights.append(weight)
    else:
      portfolio_weights.append(weight)
  portfolio_weights_array = np.array(portfolio_weights)

  return np.cumsum(np.sum(portfolio_weights_array * returns, axis=1), axis=0)