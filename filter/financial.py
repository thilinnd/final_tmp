"""
Financial filtering module for stock selection in arbitrage strategy
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


class StockFilter:
    """
    Class for filtering and selecting stocks for arbitrage strategy
    """
    
    def __init__(self, correlation_threshold: float = 0.6, 
                 min_trading_days: int = 45, 
                 adf_significance: float = 0.05):
        """
        Initialize the stock filter
        
        Args:
            correlation_threshold (float): Minimum correlation threshold for stock pairs
            min_trading_days (int): Minimum trading days required
            adf_significance (float): ADF test significance level
        """
        self.correlation_threshold = correlation_threshold
        self.min_trading_days = min_trading_days
        self.adf_significance = adf_significance
        
        # Fixed 5 stocks for VN10 arbitrage
        self.fixed_stocks = ["VIC", "VCB", "VHM", "VNM", "BID"]
    
    def select_arbitrage_stocks(self, stock_data: pd.DataFrame, 
                               futures_data: pd.Series) -> List[str]:
        """
        Select stocks for arbitrage strategy
        
        Args:
            stock_data (pd.DataFrame): Stock price data
            futures_data (pd.Series): Futures price data
            
        Returns:
            List[str]: Selected stock symbols
        """
        # For this strategy, we use fixed 5 stocks
        # In a more sophisticated implementation, you could add filtering logic here
        return self.fixed_stocks
    
    def calculate_correlation_matrix(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for stock prices
        
        Args:
            price_data (pd.DataFrame): Stock price data
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        return price_data.corr()
    
    def find_highly_correlated_pairs(self, correlation_matrix: pd.DataFrame, 
                                   threshold: float = None) -> List[Tuple[str, str, float]]:
        """
        Find highly correlated stock pairs
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            threshold (float): Correlation threshold (uses class default if None)
            
        Returns:
            List[Tuple[str, str, float]]: List of (stock1, stock2, correlation) tuples
        """
        if threshold is None:
            threshold = self.correlation_threshold
            
        pairs = []
        n = len(correlation_matrix.columns)
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = correlation_matrix.columns[i]
                stock2 = correlation_matrix.columns[j]
                corr = correlation_matrix.iloc[i, j]
                
                if abs(corr) >= threshold:
                    pairs.append((stock1, stock2, corr))
        
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    
    def test_stationarity(self, series: pd.Series) -> Dict[str, float]:
        """
        Test stationarity of a time series using ADF test
        
        Args:
            series (pd.Series): Time series to test
            
        Returns:
            Dict[str, float]: ADF test results
        """
        try:
            adf_result = adfuller(series.dropna())
            return {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < self.adf_significance
            }
        except Exception as e:
            print(f"Error in ADF test: {e}")
            return {
                'adf_statistic': np.nan,
                'p_value': 1.0,
                'critical_values': {},
                'is_stationary': False
            }
    
    def test_normality(self, series: pd.Series) -> Dict[str, float]:
        """
        Test normality of a time series using Jarque-Bera test
        
        Args:
            series (pd.Series): Time series to test
            
        Returns:
            Dict[str, float]: JB test results
        """
        try:
            jb_stat, jb_pvalue = jarque_bera(series.dropna())
            return {
                'jb_statistic': jb_stat,
                'p_value': jb_pvalue,
                'is_normal': jb_pvalue > 0.05
            }
        except Exception as e:
            print(f"Error in JB test: {e}")
            return {
                'jb_statistic': np.nan,
                'p_value': 0.0,
                'is_normal': False
            }
    
    def calculate_spread_statistics(self, spread: pd.Series) -> Dict[str, float]:
        """
        Calculate spread statistics for pair trading
        
        Args:
            spread (pd.Series): Spread time series
            
        Returns:
            Dict[str, float]: Spread statistics
        """
        spread_clean = spread.dropna()
        
        if len(spread_clean) < self.min_trading_days:
            return {
                'mean': np.nan,
                'std': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'min': np.nan,
                'max': np.nan,
                'sufficient_data': False
            }
        
        return {
            'mean': spread_clean.mean(),
            'std': spread_clean.std(),
            'skewness': spread_clean.skew(),
            'kurtosis': spread_clean.kurtosis(),
            'min': spread_clean.min(),
            'max': spread_clean.max(),
            'sufficient_data': True
        }
    
    def filter_stocks_by_volume(self, stock_data: pd.DataFrame, 
                               volume_data: pd.DataFrame = None,
                               min_avg_volume: float = 1000000) -> List[str]:
        """
        Filter stocks by trading volume (if volume data available)
        
        Args:
            stock_data (pd.DataFrame): Stock price data
            volume_data (pd.DataFrame): Volume data (optional)
            min_avg_volume (float): Minimum average volume threshold
            
        Returns:
            List[str]: Filtered stock symbols
        """
        if volume_data is None:
            # If no volume data, return all stocks
            return stock_data.columns.tolist()
        
        # Calculate average volume for each stock
        avg_volumes = volume_data.mean()
        high_volume_stocks = avg_volumes[avg_volumes >= min_avg_volume].index.tolist()
        
        return [stock for stock in stock_data.columns if stock in high_volume_stocks]
    
    def get_stock_quality_score(self, stock_data: pd.DataFrame, 
                               stock_symbol: str) -> float:
        """
        Calculate quality score for a stock based on various metrics
        
        Args:
            stock_data (pd.DataFrame): Stock price data
            stock_symbol (str): Stock symbol to evaluate
            
        Returns:
            float: Quality score (0-1, higher is better)
        """
        if stock_symbol not in stock_data.columns:
            return 0.0
        
        stock_series = stock_data[stock_symbol].dropna()
        
        if len(stock_series) < self.min_trading_days:
            return 0.0
        
        # Calculate various quality metrics
        returns = stock_series.pct_change().dropna()
        
        # Volatility (lower is better for arbitrage)
        volatility = returns.std() * np.sqrt(252)
        volatility_score = max(0, 1 - volatility / 0.5)  # Normalize to 0-1
        
        # Consistency (lower variance in returns is better)
        return_variance = returns.var()
        consistency_score = max(0, 1 - return_variance / 0.01)  # Normalize to 0-1
        
        # Data completeness
        completeness_score = len(stock_series) / len(stock_data)
        
        # Overall quality score (weighted average)
        quality_score = (
            0.4 * volatility_score +
            0.3 * consistency_score +
            0.3 * completeness_score
        )
        
        return min(1.0, max(0.0, quality_score))
    
    def select_best_stocks(self, stock_data: pd.DataFrame, 
                          n_stocks: int = 5) -> List[str]:
        """
        Select the best n stocks for arbitrage based on quality scores
        
        Args:
            stock_data (pd.DataFrame): Stock price data
            n_stocks (int): Number of stocks to select
            
        Returns:
            List[str]: Selected stock symbols
        """
        # For this implementation, we use the fixed 5 stocks
        # In a more sophisticated version, you could implement quality-based selection
        available_stocks = [stock for stock in self.fixed_stocks 
                           if stock in stock_data.columns]
        
        return available_stocks[:n_stocks]


def create_stock_filter(correlation_threshold: float = 0.6,
                       min_trading_days: int = 45,
                       adf_significance: float = 0.05) -> StockFilter:
    """
    Factory function to create a StockFilter instance
    
    Args:
        correlation_threshold (float): Minimum correlation threshold
        min_trading_days (int): Minimum trading days required
        adf_significance (float): ADF test significance level
        
    Returns:
        StockFilter: Configured stock filter instance
    """
    return StockFilter(
        correlation_threshold=correlation_threshold,
        min_trading_days=min_trading_days,
        adf_significance=adf_significance
    )
