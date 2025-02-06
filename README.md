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
### Attention-based Auto-encoder for Anomaly Detection  

Anomaly detection in cryptocurrency markets requires an adaptive model capable of capturing complex patterns in high-dimensional data. We employ an **attention-based auto-encoder**, which leverages self-attention mechanisms to focus on the most relevant features in the data. This enhances the model’s ability to detect deviations from normal price behavior.  

#### Mathematical Formulation  

Let $\mathbf{Y}_t \equiv \mathbf{X} \in \mathbb{R}^{d\times n}$ be the data matrix at time step $t$, where:  
- $d$ is the number of feature inputs (on-chain indicators and price-related metrics).  
- $n$ is the number of cryptocurrencies being analyzed.  

We define the learnable **weight matrices** for the attention mechanism:  

$$
\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d\times d}
$$

which transform the input matrix $\mathbf{X}$ into the **query**, **key**, and **value** representations: 

$$
\mathbf{Q} = \mathbf{W}_Q\mathbf{X}, \quad \mathbf{K} = \mathbf{W}_K\mathbf{X}, \quad \mathbf{V} = \mathbf{W}_V\mathbf{X}
$$

where $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{d\times n}$.  

To compute the **attention matrix**, we apply the scaled dot-product attention mechanism:  

$$
\mathbf{Z} = \text{softmax} \left(\frac{\mathbf{QK}^\top}{\sqrt{d}}\right)
$$

which ensures numerical stability by normalizing the attention scores. The **weighted feature representation** is then given by:  

$$
\mathbf{A} = \mathbf{ZV}
$$

where $\mathbf{Z} \in \mathbb{R}^{d\times d}$ and $\mathbf{A} \in \mathbb{R}^{d\times n}$ represent the refined feature space, emphasizing relevant patterns for anomaly detection.  

#### Latent Space Representation and Reconstruction  

To project the data into a lower-dimensional latent space, we introduce a learnable **context vector** $\mathbf{C} \in \mathbb{R}^d$ and weight matrix $\mathbf{W}_l \in \mathbb{R}^{n\times l}$ with $l < n$:

$$
\mathbf{S} = \mathbf{C}^\top \mathbf{A}, \quad \mathbf{L} = \text{ReLU}(\mathbf{S} \mathbf{W}_l)
$$
where $\mathbf{L}$ represents the compressed feature encoding. Finally, the reconstruction step is performed using the weight matrix $\mathbf{W} \in \mathbb{R}^{l\times n}$, producing the estimated cryptocurrency return vector:  
$$
\hat{\mathbf{R}_t} = \mathbf{L} \mathbf{W}
$$
where $\hat{\mathbf{R}}_t \in \mathbb{R}^{n}$ represents the reconstructed returns.  

This **attention-enhanced auto-encoder** effectively isolates **irregular market behaviors** by comparing actual and reconstructed price movements, identifying anomalies that deviate from learned market dynamics.  

### On-Chain Indicators Data

  1. Market-Value-to-Realized-Value (MVRV): Identifies overvaluation/undervaluation.
   
    The MVRV ratio compares a cryptocurrency’s market capitalization to its realized capitalization, showing whether the asset is overvalued or undervalued. A high MVRV suggests that holders have large unrealized profits, increasing the risk of profit-taking and price corrections. A low MVRV indicates potential buying opportunities as the asset may be undervalued. This metric helps traders identify market tops and bottoms for better entry and exit decisions. The MVRV ratio is defined as: 
    
  $$MVRV = \frac{\text{Market Value}}{\text{Realized Value}}$$

  where **Market value (Market Capitalization)** is the total market worth of all circulating coins 
  $$\text{Market Value} = \text{Price} \times \text{Circulating Supply}$$
  
  and Realized Value is the sum of all coins valued at their last moved price: 
  $$\text{Realized Value} = \sum_{i=1}^{N} \text{Price}_i \times \text{Coins}_{i}$$

  1. Network-Value-to-Transaction (NVT) Ratio: Assesses network activity and speculative behavior.

    The NVT ratio compares a cryptocurrency’s market value to its transaction volume, similar to the P/E ratio in stocks. A high NVT suggests overvaluation or speculation, while a low NVT indicates strong network fundamentals and a healthier valuation. This ratio helps detect bubbles and undervaluation, providing insights beyond price movements for better trading strategies. The NVT ratio is given by:
  
  $$NVT = \frac{\text{Market Value}}{\text{Transaction Volume}}$$

  where **Transaction Volume** is the total on-chain transaction value over a given period.

### Data Setup  

We define the key financial indicators for each cryptocurrency at time $t$:  
- $P_t$: Price of the cryptocurrency.  
- $M_t$: Market Value to Realized Value (MVRV) ratio.  
- $V_t$: Network Value to Transactions (NVT) ratio.  

#### Feature Construction  

To capture the market dynamics, we construct three main features:  

1. **Log-returns of the cryptocurrency at time** $t$:  
   $$R_t = \log\left(\frac{P_t}{P_{t-1}}\right)$$  
   This measures the relative price change, commonly used in financial modeling.  

2. **Differenced log-MVRV of the cryptocurrency at time** $t$:  
   $$M'_t = \log\left(\frac{M_t}{M_{t-1}}\right)$$  
   This represents the percentage change in the MVRV ratio, reflecting valuation shifts.  

3. **Differenced log-NVT of the cryptocurrency at time** $t$:  
   $$V'_t = \log\left(\frac{V_t}{V_{t-1}}\right)$$  
   This tracks the relative change in the NVT ratio, indicating transaction activity variations.  

#### Data Representation  

At each time step $t$, we construct the feature matrix **$X_t$** by organizing these indicators for multiple cryptocurrencies.  
For $n$ cryptocurrencies, we define:  

- **Log-returns vector**: $\mathbf{R}_t = [R_1, R_2, \dots, R_n]_t$  
- **Differenced log-MVRV vector**: $\mathbf{M}'_t = [M'_1, M'_2, \dots, M'_n]_t$  
- **Differenced log-NVT vector**: $\mathbf{V}'_t = [V'_1, V'_2, \dots, V'_n]_t$  

The full feature matrix is then structured as:  
$$
X_t = 
\begin{bmatrix}
R_1 & R_2 & \dots & R_n \\
M'_1 & M'_2 & \dots & M'_n \\
V'_1 & V'_2 & \dots & V'_n
\end{bmatrix}_t \in \mathbb{R}^{3\times n}
$$  
where $n$ represents the number of cryptocurrencies in the dataset.  

#### Attention Auto-encoder Mapping  

The **attention-based auto-encoder** model, denoted as $\Phi$, is designed to learn meaningful representations from the feature matrix and reconstruct expected returns:  
$$
\Phi: X_t \rightarrow \hat{\mathbf{R}}_t
$$  
where $\hat{\mathbf{R}}_t$ is the predicted return vector. By leveraging attention mechanisms, this model captures complex dependencies between asset returns and blockchain-based indicators, enhancing anomaly detection capabilities.  


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
