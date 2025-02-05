# Anomaly Detection in the Cryptocurrency Market Using On-Chain Data

## Abstract

This study addresses the problem of anomaly detection in the cryptocurrency market using on-chain data to enhance trading strategies. On-chain data, which includes verified transactions recorded on the blockchain, contains critical insights into market behavior. This research applies an attention-based Auto-encoder model, a powerful tool for anomaly detection, to capture subtle patterns in cryptocurrency price returns. We focus on key on-chain indicators such as the Market-Value-to-Realized-Value (MVRV) and the Network-Value-to-Transaction (NVT) ratio, which are widely used to assess market conditions and potential bubbles. The attention mechanism enhances the Auto-encoder’s ability to focus on the most relevant features in these complex datasets, allowing for more accurate identification of abnormal market behaviors. By detecting these anomalies, the model aims to provide actionable insights that can be leveraged in real-time trading strategies, helping market participants optimize their decision-making and potentially profit from price shifts driven by hidden market forces.


---

## Introduction

Cryptocurrency markets are known for their high volatility, with extreme price fluctuations occurring frequently. While this volatility poses significant risks, it also creates opportunities for traders and investors. Price movements often lead to temporary undervaluation or overvaluation when compared to on-chain data, offering potential profit-making opportunities.

On-chain data consists of blockchain-recorded transaction metrics, providing insights similar to financial ratios in traditional markets. However, unlike traditional financial indicators, on-chain data reflects real-time network activity by analyzing transaction flows, wallet movements, and liquidity metrics within the blockchain. Two widely used on-chain indicators for market valuation are the Market-Value-to-Realized-Value (MVRV) ratio and the Network-Value-to-Transaction (NVT) ratio.

The MVRV ratio compares a cryptocurrency’s market capitalization to its realized capitalization, which represents the aggregate cost basis of all coins in circulation. A high MVRV suggests that the market is overvalued, potentially signaling a price correction, while a low MVRV indicates undervaluation, suggesting a possible buying opportunity.

The NVT ratio, on the other hand, measures a cryptocurrency’s network value relative to transaction volume. It functions similarly to the Price-to-Earnings (P/E) ratio in traditional markets, where a high NVT suggests that the network is overvalued relative to transaction activity, potentially indicating a speculative bubble. A low NVT implies strong network fundamentals and healthy transaction activity.

Historical patterns suggest that cryptocurrency price returns often align with on-chain indicators such as MVRV and NVT. However, there are instances when price movements deviate from expected trends, exhibiting irregular or anomalous behavior. Detecting these anomalies can enhance trading strategies and risk management, helping market participants identify inefficiencies and capitalize on hidden market signals.

Traditional statistical methods and rule-based approaches often struggle to detect anomalies in cryptocurrency markets due to their high volatility and complex, nonlinear price behavior. To address this challenge, we employ an attention-based Auto-Encoder, a deep learning model designed to learn the underlying patterns of normal market conditions and identify deviations that indicate anomalies.

An Auto-Encoder consists of an encoder, which compresses input data into a lower-dimensional representation, and a decoder, which reconstructs the original data. By training the model on historical on-chain and price data, the Auto-Encoder learns to minimize reconstruction error for typical market behaviors. However, when presented with anomalous market conditions, the reconstruction error increases significantly, signaling a potential anomaly.

The attention mechanism enhances the Auto-Encoder’s ability to focus on the most informative features within the data, improving its accuracy in detecting subtle irregularities. Given the complexity of cryptocurrency markets, the attention-based Auto-Encoder enables us to identify unexpected price movements that deviate from historical patterns and on-chain fundamentals, making it a powerful tool for refining trading strategies and risk management.

---

## Methodology
- Model: Attention-based Auto-encoder for anomaly detection.

Let $\mathbf{Y}_t\equiv \mathbf{X}\in \mathbb{R}^{d\times n}$ be the data point matrix given observation $t$ where $d$ is the number of feature inputs and $n$ is the number of currencies.

Let $\mathbf{W}_Q,\ \mathbf{W}_K, \mathbf{W}_V\in \mathbb{R}^{d\times d}$ be the learnable parameter matrices.

The autoencoder model follow the chain of equations:

$$\mathbf{Q} = \mathbf{W}_Q\mathbf{X},\quad \mathbf{K} = \mathbf{W}_K\mathbf{X},\quad \mathbf{V} = \mathbf{W}_V\mathbf{X}$$

with $\mathbf{Q,\ W,\ V}\in \mathbb{R}^{d\times n}$
- Key Features:
  - Captures complex patterns in cryptocurrency price returns.
  - Uses an attention mechanism to prioritize important features.
- On-Chain Indicators Used:
  - Market-Value-to-Realized-Value (MVRV): Identifies overvaluation/undervaluation.
  - Network-Value-to-Transaction (NVT) Ratio: Assesses network activity and speculative behavior.

---

## Results & Insights 

- The attention mechanism improves the model’s ability to detect abnormal market behaviors.
- Early anomaly detection provides real-time trading signals for risk mitigation and profit opportunities.
- The approach enhances decision-making by revealing patterns not visible through traditional price analysis.

---

## Conclusion & Future Work 
- Key Contribution: A novel anomaly detection method leveraging on-chain data + deep learning.
- Real-World Impact: Can be used by traders and investors to anticipate market shifts.
- Next Steps: Testing on additional on-chain metrics and real-time deployment for automated trading.

---

## Visuals & Data Representations 📈
- Diagram of the Attention-based Auto-encoder architecture.
- Example of detected anomalous market behaviors with MVRV & NVT trends.
- Performance comparison of the model against traditional methods.
