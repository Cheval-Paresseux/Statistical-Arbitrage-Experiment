# Cointegration Theory

This branch implements a mean-reversion trading strategy based on cointegration theory. The approach focuses on identifying pairs or groups of assets whose prices exhibit a long-term equilibrium relationship. When their price spread deviates from this equilibrium, the strategy exploits the mean-reverting behavior by taking trading positions.

## Files and Notebooks

### Python Scripts

- **`ClusterMethods.py`**: Implements functions for clustering assets based on historical price data to identify potential cointegrated pairs.
- **`CombinationMethods.py`**: Provides methods for forming asset combinations from the clusters, focusing on finding cointegrated pairs or groups.
- **`SignalsMethods.py`**: Contains logic for generating trading signals by analyzing out-of-sample data.

### Jupyter Notebook

- **Cointegration Strategy Simulation**: A notebook that simulates the entire strategy on a restricted universe of S&P500 assets. It includes clustering, cointegration testing, and applying mean-reversion signals for trading.

## Strategy Workflow

1. **Clustering**: Assets are grouped into clusters with similar price behavior using the methods in `ClusterMethods.py`.
2. **Combination Creation**: Cointegrated asset combinations are identified from the clusters using `CombinationMethods.py`.
3. **Signal Generation**: Based on the deviations from the mean price spread, trading signals are generated using `SignalsMethods.py`.
