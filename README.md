# Supply Chain Forecasting with Deep Learning

## Overview

This repository demonstrates a comparative study of Machine Learning and Deep Learning models applied to supply chain demand forecasting. The goal is to:

1. **Forecast future sales** by understanding historical sales trends.
2. **Optimize inventory** to match store stock with actual demand and reduce storage costs.
3. **Optimize replenishment** by determining ideal order quantities, lowering warehousing and transportation expenses.
4. **Identify key sales drivers** (e.g., promotions, seasonality) to enhance business strategies.

## Problem Statement

- **Sales Trend Forecasting:** Learn from past data to predict future demand.
- **Inventory Optimization:** Balance stock levels to minimize excess and shortages.
- **Replenishment Optimization:** Compute optimal order sizes and frequencies.
- **Sales Driver Analysis:** Uncover features that impact sales volume.

## Models Compared

### Machine Learning

- **Linear Regression**: Fits a straight line to data; fast and interpretable but limited to linear patterns.
- **XGBoost**: Gradient-boosted decision trees with built-in regularization, parallel split finding, and native handling of missing values.

### Deep Learning

- **CNN (Convolutional Neural Network)**: Extracts local temporal features in time-series via 1D convolutions.
- **LSTM (Long Short-Term Memory)**: Captures long-term dependencies in sequences with memory cells.
- **CNN + LSTM**: Hybrid model combining CNN’s feature extraction with LSTM’s sequence modeling.
- **GRU (Gated Recurrent Unit)**: Simplified RNN unit addressing vanishing gradients with fewer parameters.
- **Transformers**: Self-attention architecture for modeling long-range dependencies without recurrence.

## Key Findings

- **XGBoost** achieved the best performance among classical Machine Learning models with minimal hyperparameter tuning.
- The **CNN + LSTM** hybrid model yielded the highest forecast accuracy among Deep Learning approaches.

## Advantages

### XGBoost

1. **Out-of-the-box performance** with default settings.
2. **Handles sparse and missing data** natively.
3. **Regularized objective** to reduce overfitting.
4. **Parallel training** speeds up model building.
5. **Feature importance** for interpretability.

### CNN + LSTM

1. **Convolutional layers** learn local temporal patterns automatically.
2. **Pooling layers** reduce dimensionality while retaining key information.
3. **Flatten + RepeatVector** transforms convolutional outputs for LSTM input.
4. **Dense output layer** integrates learned features into forecasts.

## Getting Started

### Prerequisites

- Python 3.7+
- Install required packages:
  ```bash
  pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
