import pandas as pd
import numpy as np
import riskfolio as rp
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# This file objective is to store the class for clustering methods

# As testing every possible combination takes too much time, we reduce our universe of assets by defining clusters.
#   1. Each asset history is transformed into the exp(sum(log return)) to avoid scale biases
#   2. Chose a method to clusterize.

# The results of cluterization methods should always be : list of Dataframes with asset exp(log return)) history


# ------------------------------------------------------------------------------------------------------------------------------------------------


class Clustering:
    """
    A class to perform filtering operations on a DataFrame of assets.
    The objective is to create clusters of assets that are likely to move together.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the Filtering class with a DataFrame.

        Args:
        df (pd.DataFrame): The DataFrame containing asset data.
        """
        self.df = df

    def riskfolio_filter(
        self, linkage: str, opt_k_methods: str, plot: str, kind: str = "spring"
    ):
        """
        Performs clustering on the asset log returns.

        Args:
        linkage (str): Linkage method for clustering.
                -> linkages = ['single','complete','average','weighted','centroid', 'median', 'ward','DBHT']
        opt_k_methods (str): Method to optimize the number of clusters
                -> opt_k_methods = ['twodiff', 'stdsil']
        plot (str): Type of plot to generate ('Dendrogram' or 'Network').
                -> plot = ['Dendrogram' or 'Network']
        kind (str): Layout for the network plot if 'Network' is selected. Default is 'spring'.
                -> kind = ['spring','kamada','planar','circular']

        Returns:
        clusters (list): List containing the DataFrames of assets for each cluster. No transformation on assets prices on return.
        """
        # 1) Transforming the data to avoid scale biases
        log_prices = np.log(self.df)

        # 2) Performing the clusterization and plot the results
        # Using Dendrogram graph
        if plot == "Dendrogram":
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            rp.plot_dendrogram(
                returns=log_prices,
                codependence="pearson",
                linkage=linkage,
                opt_k_method=opt_k_methods,
                k=None,
                max_k=10,
                leaf_order=True,
                ax=ax,
            )
            plt.show()
            clusters = rp.assets_clusters(
                returns=log_prices,
                codependence="pearson",
                linkage=linkage,
                k=None,
                max_k=10,
                leaf_order=True,
            )
        # Using Network Graph
        elif plot == "Network":
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            rp.plot_network(
                returns=log_prices,
                codependence="pearson",
                linkage=linkage,
                k=None,
                max_k=10,
                leaf_order=True,
                kind=kind,
                seed=0,
                ax=ax,
            )
            plt.show()
            clusters = rp.assets_clusters(
                returns=log_prices,
                codependence="pearson",
                linkage=linkage,
                k=None,
                max_k=10,
                leaf_order=True,
            )

        else:
            raise ValueError("Invalid plot type. Choose 'Dendrogram' or 'Network'.")

        # 3) Preparing data before returning
        clusters_df = pd.DataFrame(clusters)
        clusters_df.reset_index(drop=True, inplace=True)

        # Putting the result as a dictionnary of Dataframes (key: cluster)
        liste = {}
        for index, cluster in clusters_df["Clusters"].items():
            if cluster not in liste:
                liste[cluster] = [clusters_df["Assets"][index]]
            else:
                liste[cluster].append(clusters_df["Assets"][index])

        dataframes_clusters = []

        for cluster_name, tickers in liste.items():
            colonnes_cluster = [
                ticker for ticker in tickers if ticker in self.df.columns
            ]
            df_cluster = self.df[colonnes_cluster]
            dataframes_clusters.append(df_cluster)

        return dataframes_clusters

    def dtw_filter(self, n_clusters):
        """
        This method uses Distance Time Warping to clusterize assets on their log prices.
        We usually choose to compute the same number of clusters as the number of combinations we want to trade simultaneously.
        For example : I want to trade 5 combiantions at a time, then i will choose each combination from a different cluster for mrket neutrality.
        Args:
            n-clusters (int) -> represent the number of clusters we want to generate.

        Returns:
            dataframes_clusters (list) -> each dataframe in this list corresponds to the assets of a cluster. No transformation on assets prices on return.
        """

        # Transpose the dataframe so that each row represents a time series
        log_df = np.log(self.df)
        time_series_data = log_df.T.values

        # Scale the time series data
        scaler = TimeSeriesScalerMeanVariance()
        time_series_data = scaler.fit_transform(time_series_data)

        # Apply TimeSeriesKMeans clustering with DTW metric
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
        labels = model.fit_predict(time_series_data)

        # Create a list to hold the resulting dataframes for each cluster
        dataframes_clusters = []

        for cluster in range(n_clusters):
            # Select columns belonging to the current cluster
            cluster_columns = self.df.columns[labels == cluster]
            dataframes_clusters.append(self.df[cluster_columns])

        return dataframes_clusters
