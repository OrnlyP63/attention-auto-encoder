import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Set Tahoma font and style
plt.rcParams["font.family"] = "Tahoma"
sns.set_style("whitegrid")

def plot(df:pd.DataFrame, ticker:str):
  name = df['asset'].iloc[0]
  fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)  # 3 rows, 1 column
  for col in ['CapMVRVCur', 'NVTAdj', 'returns']:
    sns.lineplot(x="time", y=col, data=df, ax=axes[0], color="royalblue", label=col + f' of {name}')

  # Set common x-axis label and title
  axes[-1].set_xlabel("Date", fontsize=14)
  fig.suptitle(f"{name} On-Chain Data", fontsize=16)

  # Rotate x-axis labels for better readability
  plt.xticks(rotation=45)

  # Adjust layout
  plt.tight_layout()
  plt.show()
    