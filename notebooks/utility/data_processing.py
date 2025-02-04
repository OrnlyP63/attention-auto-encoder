import pandas as pd
import numpy as np

def set_time_data(currency_metrics: pd.DataFrame):
  metrics = currency_metrics.copy()
  metrics['time'] = pd.to_datetime(metrics['time'])
  metrics = metrics.set_index('time')

  return metrics

def set_asset_return(currency_metrics: pd.DataFrame):
  metrics = currency_metrics.copy()
  if 'returns' in metrics.columns:
    return metrics
  else:
    log_prices = np.log(metrics['PriceUSD'])
    metrics['returns'] = log_prices.diff()
    return metrics.dropna()