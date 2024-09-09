import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# This file objective is to gather methods for getting Signals

# ------------------------------------------------------------------------------------------------------------------------------------------------


def dynamic_ou_signals(
    comb_dict: dict,
    entry_threshold: int,
    exit_threshold: int,
    stoploss_threshold: int,
    rolling_adf_threshold: float,
) -> pd.DataFrame:
    """ """
    weights = comb_dict["weights"]
    cluster = comb_dict["cluster"]
    df = comb_dict["df"]

    # Create a copy of the input dataframe
    df_copy = df.copy()

    # Initialize the 'signals' column with NaN or a default value
    df_copy["signals"] = pd.NA

    for idx in df_copy.index:
        # We manage in sample data with this simple trick
        adf_pvalue = df_copy.loc[idx, "adf_pvalue"]
        if adf_pvalue == 0:
            new_status = 0
        elif adf_pvalue < rolling_adf_threshold:
            z = df_copy.loc[idx, "z_score"]
            # Use .get() to avoid KeyError and handle missing values
            last_status = (
                df_copy.loc[idx - timedelta(days=1), "signals"]
                if (idx - timedelta(days=1)) in df_copy.index
                else 0
            )
            new_status = last_status

            # Check for entry points
            if (
                (last_status == 0)
                and (z > entry_threshold)
                and (z < stoploss_threshold)
            ):
                new_status = -1
            elif (
                (last_status == 0)
                and (z < -entry_threshold)
                and (z > -stoploss_threshold)
            ):
                new_status = 1

            # Check for exit points
            elif ((last_status == -1) and (z < exit_threshold)) or (
                (last_status == -1) and (z > stoploss_threshold)
            ):
                new_status = 0
            elif ((last_status == 1) and (z > -exit_threshold)) or (
                (last_status == 1) and (z < -stoploss_threshold)
            ):
                new_status = 0
        elif adf_pvalue > rolling_adf_threshold:
            new_status = 0

        df_copy.loc[idx, "signals"] = new_status

    results = {"weights": weights, "cluster": cluster, "df": df_copy}

    return results


# ------------------------------------------------------------------------------------------------------------------------------------------------


def plot_price_and_signals(
    signals_df: pd.DataFrame,
    nbr_assets: int,
    entry_threhsold: int,
    stoploss_threhsold: int,
):
    # Define color map for signals
    signal_colors = {0: "blue", 1: "green", -1: "red"}

    plt.figure(figsize=(10, 5))

    # Plot residuals
    plt.plot(
        signals_df["residuals"].index,
        signals_df["residuals"],
        label="Residuals",
        color="black",
        linewidth=2,
    )

    # Plot moving average
    plt.plot(
        signals_df["moving_mean"].index,
        signals_df["moving_mean"],
        label="Moving Average",
        color="purple",
        linestyle="--",
        linewidth=2,
    )

    # Plot moving average + std
    plt.plot(
        signals_df["moving_mean"].index,
        signals_df["moving_mean"] + entry_threhsold * signals_df["moving_sigma"],
        label="Short entry point",
        color="red",
        linestyle="--",
        linewidth=1.5,
    )

    # Plot moving average - std
    plt.plot(
        signals_df["moving_mean"].index,
        signals_df["moving_mean"] - entry_threhsold * signals_df["moving_sigma"],
        label="Long entry point",
        color="green",
        linestyle="--",
        linewidth=1.5,
    )

    # Plot moving average + std
    plt.plot(
        signals_df["moving_mean"].index,
        signals_df["moving_mean"] + stoploss_threhsold * signals_df["moving_sigma"],
        label="Short stoploss exit point",
        color="red",
        linestyle="--",
        linewidth=1.5,
    )

    # Plot moving average - std
    plt.plot(
        signals_df["moving_mean"].index,
        signals_df["moving_mean"] - stoploss_threhsold * signals_df["moving_sigma"],
        label="Long stoploss exit point",
        color="green",
        linestyle="--",
        linewidth=1.5,
    )

    # Mark signals with corresponding colors
    for status_value in signals_df["signals"].unique():
        mask = signals_df["signals"] == status_value
        plt.scatter(
            signals_df.index[mask],
            signals_df["residuals"][mask],
            label=f"Status {status_value}",
            color=signal_colors[status_value],
            marker="o",
            s=50,
        )

    assets_name = signals_df.columns[:nbr_assets].tolist()
    plt.title(f"Residuals, Moving Average, and Bands for {assets_name}")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------------------------------
