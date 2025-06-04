#================================================================
#
#   File name   : indicators.py
#   Author      : PyLessons
#   Created date: 2021-02-25
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : Used to plot 5 indicators with OHCL bars and etc.
#
#================================================================
"""
This module provides functions to calculate and add various technical indicators
to a Pandas DataFrame for financial analysis, specifically for a Bitcoin trading bot.

The main function, `AddIndicators`, takes a DataFrame with OHLCV (Open, High, Low, Close, Volume)
data and adds columns for several common technical indicators. Other functions provide
utilities for feature selection based on correlation and for grouping indicators.
"""
import pandas as pd
from ta.trend import SMAIndicator, macd, PSARIndicator
from ta.volatility import BollingerBands
from ta.momentum import rsi
from ta.volume import VolumeWeightedAveragePrice
from utils import Plot_OHCL
from ta import add_all_ta_features, add_trend_ta, add_volume_ta, add_volatility_ta, add_momentum_ta, add_others_ta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def AddIndicators(df):
    """
    Adds various technical indicators to the input DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with at least "High", "Low", "Close", "Volume" columns.

    Returns:
        pd.DataFrame: DataFrame with added indicator columns.
    """

    # Simple Moving Average (SMA)
    # Calculates the average of a selected range of prices, usually closing prices, by the number of periods in that range.
    # Parameters:
    #   close (pd.Series): Closing price series.
    #   window (int): Time period (number of bars) for the SMA.
    #   fillna (bool): If True, fill NaN values.
    df["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    df["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()
    
    # Bollinger Bands (BB)
    # A volatility indicator that consists of a middle band (SMA) and two outer bands (standard deviations away from the SMA).
    # Parameters:
    #   close (pd.Series): Closing price series.
    #   window (int): Time period for the SMA (middle band). Default is 20.
    #   window_dev (int): Number of standard deviations for the outer bands. Default is 2.
    indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()  # Middle Band
    df['bb_bbh'] = indicator_bb.bollinger_hband()  # Upper Band
    df['bb_bbl'] = indicator_bb.bollinger_lband()  # Lower Band

    # Parabolic Stop and Reverse (Parabolic SAR)
    # A trend-following indicator used to determine potential stop and reversal points.
    # Parameters:
    #   high (pd.Series): High price series.
    #   low (pd.Series): Low price series.
    #   close (pd.Series): Closing price series.
    #   step (float): Acceleration factor. Default is 0.02.
    #   max_step (float): Maximum acceleration factor. Default is 0.2.
    #   fillna (bool): If True, fill NaN values.
    indicator_psar = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=0.02, max_step=0.2, fillna=True) # max_step was 2, changed to common 0.2
    df['psar'] = indicator_psar.psar()

    # Moving Average Convergence Divergence (MACD)
    # A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
    # Parameters: (Commented out in original code)
    #   close (pd.Series): Closing price series.
    #   window_slow (int): Time period for the slow EMA. Default is 26.
    #   window_fast (int): Time period for the fast EMA. Default is 12.
    #   window_sign (int): Time period for the signal line EMA. Default is 9.
    #   fillna (bool): If True, fill NaN values.
    # df["MACD"] = macd(close=df["Close"], window_slow=26, window_fast=12, fillna=True) # mazas (original comment)

    # Relative Strength Index (RSI)
    # A momentum oscillator that measures the speed and change of price movements.
    # Values range from 0 to 100. Typically, RSI above 70 indicates overbought conditions, and below 30 indicates oversold conditions.
    # Parameters:
    #   close (pd.Series): Closing price series.
    #   window (int): Time period for RSI calculation. Default is 14.
    #   fillna (bool): If True, fill NaN values.
    df["RSI"] = rsi(close=df["Close"], window=14, fillna=True) # mazas (original comment)

    # Volume Weighted Average Price (VWAP)
    # The average price a security has traded at throughout the day, based on both volume and price.
    # It is important because it provides traders with insight into both the trend and value of a security.
    # Parameters:
    #   high (pd.Series): High price series.
    #   low (pd.Series): Low price series.
    #   close (pd.Series): Closing price series.
    #   volume (pd.Series): Volume series.
    #   window (int): Time period for VWAP calculation. Default is 14. (Used 14 in previous commit)
    #   fillna (bool): If True, fill NaN values.
    df["vwap"] = VolumeWeightedAveragePrice(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=14, fillna=True).volume_weighted_average_price()
    
    return df

def DropCorrelatedFeatures(df, threshold, plot):
    df_copy = df.copy()

    # Remove OHCL columns
    df_drop = df_copy.drop(["Date", "Open", "High", "Low", "Close", "Volume"], axis=1)

    # Calculate Pierson correlation
    df_corr = df_drop.corr()

    columns = np.full((df_corr.shape[0],), True, dtype=bool)
    for i in range(df_corr.shape[0]):
        for j in range(i+1, df_corr.shape[0]):
            if df_corr.iloc[i,j] >= threshold or df_corr.iloc[i,j] <= -threshold:
                if columns[j]:
                    columns[j] = False
                    
    selected_columns = df_drop.columns[columns]

    df_dropped = df_drop[selected_columns]

    if plot:
        # Plot Heatmap Correlation
        fig = plt.figure(figsize=(8,8))
        ax = sns.heatmap(df_dropped.corr(), annot=True, square=True)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0) 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        fig.tight_layout()
        plt.show()
    
    return df_dropped

def get_trend_indicators(df, threshold=0.5, plot=False):
    df_trend = df.copy()
    
    # add custom trend indicators
    df_trend["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df_trend["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    df_trend["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()

    df_trend = add_trend_ta(df_trend, high="High", low="Low", close="Close")

    return DropCorrelatedFeatures(df_trend, threshold, plot)

def get_volatility_indicators(df, threshold=0.5, plot=False):
    df_volatility = df.copy()
    
    # add custom volatility indicators
    # ...

    df_volatility = add_volatility_ta(df_volatility, high="High", low="Low", close="Close")

    return DropCorrelatedFeatures(df_volatility, threshold, plot)

def get_volume_indicators(df, threshold=0.5, plot=False):
    df_volume = df.copy()
    
    # add custom volume indicators
    # ...

    df_volume = add_volume_ta(df_volume, high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_volume, threshold, plot)

def get_momentum_indicators(df, threshold=0.5, plot=False):
    df_momentum = df.copy()
    
    # add custom momentum indicators
    # ...

    df_momentum = add_momentum_ta(df_momentum, high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_momentum, threshold, plot)

def get_others_indicators(df, threshold=0.5, plot=False):
    df_others = df.copy()
    
    # add custom indicators
    # ...

    df_others = add_others_ta(df_others, close="Close")

    return DropCorrelatedFeatures(df_others, threshold, plot)

def get_all_indicators(df, threshold=0.5, plot=False):
    df_all = df.copy()
    
    # add custom indicators
    # ...

    df_all = add_all_ta_features(df_all, open="Open", high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_all, threshold, plot)

def indicators_dataframe(df, threshold=0.5, plot=False):
    trend       = get_trend_indicators(df, threshold=threshold, plot=plot)
    volatility  = get_volatility_indicators(df, threshold=threshold, plot=plot)
    volume      = get_volume_indicators(df, threshold=threshold, plot=plot)
    momentum    = get_momentum_indicators(df, threshold=threshold, plot=plot)
    others      = get_others_indicators(df, threshold=threshold, plot=plot)
    #all_ind = get_all_indicators(df, threshold=threshold)

    final_df = [df, trend, volatility, volume, momentum, others]
    result = pd.concat(final_df, axis=1)

    return result

if __name__ == "__main__":   
    df = pd.read_csv('./BTCUSD_1h.csv')
    df = df.sort_values('Date')
    #df = AddIndicators(df)

    #test_df = df[-400:]

    #Plot_OHCL(df)
    get_others_indicators(df, threshold=0.5, plot=True)
