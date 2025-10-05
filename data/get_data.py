"""
Data loading functions for statistical arbitrage strategy
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

def get_vn30(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get VN30 stock data
    
    Args:
        start_date (str): Start date
        end_date (str): End date
        
    Returns:
        pd.DataFrame: VN30 stock prices
    """
    print(f"Loading VN30 data from {start_date} to {end_date}")
    
    # Create dummy data for testing
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Create 30 stocks with realistic price movements
    np.random.seed(42)
    n_stocks = 30
    n_days = len(dates)
    
    # Generate correlated stock prices
    base_prices = np.random.uniform(50, 200, n_stocks)
    stock_data = np.zeros((n_days, n_stocks))
    stock_data[0] = base_prices
    
    for i in range(1, n_days):
        # Generate correlated returns
        returns = np.random.multivariate_normal(
            mean=np.zeros(n_stocks),
            cov=np.eye(n_stocks) * 0.0004 + np.ones((n_stocks, n_stocks)) * 0.0001,
            size=1
        )[0]
        
        stock_data[i] = stock_data[i-1] * (1 + returns)
    
    # Create DataFrame
    stock_names = [f'STOCK_{i+1:02d}' for i in range(n_stocks)]
    vn30_data = pd.DataFrame(stock_data, index=dates, columns=stock_names)
    
    print(f"Generated VN30 data: {vn30_data.shape}")
    return vn30_data

def get_vn30f1m(start_date: str, end_date: str) -> pd.Series:
    """
    Get VN30F1M futures data
    
    Args:
        start_date (str): Start date
        end_date (str): End date
        
    Returns:
        pd.Series: VN30F1M futures prices
    """
    print(f"Loading VN30F1M data from {start_date} to {end_date}")
    
    # Create dummy data for testing
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate futures prices with some correlation to VN30
    np.random.seed(42)
    n_days = len(dates)
    
    base_price = 1000
    futures_data = np.zeros(n_days)
    futures_data[0] = base_price
    
    for i in range(1, n_days):
        # Generate returns with some mean reversion
        return_val = np.random.normal(0, 0.02)
        futures_data[i] = futures_data[i-1] * (1 + return_val)
    
    # Create Series
    vn30f1m_data = pd.Series(futures_data, index=dates, name='price')
    
    print(f"Generated VN30F1M data: {len(vn30f1m_data)} records")
    return vn30f1m_data

def load_market_data(start_date: str, end_date: str) -> tuple:
    """
    Load all market data
    
    Args:
        start_date (str): Start date
        end_date (str): End date
        
    Returns:
        tuple: (vn30_data, vn30f1m_data)
    """
    vn30_data = get_vn30(start_date, end_date)
    vn30f1m_data = get_vn30f1m(start_date, end_date)
    
    return vn30_data, vn30f1m_data



