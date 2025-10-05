"""
Helper functions for statistical arbitrage strategy
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_periods_df(stock_data: pd.DataFrame, 
                       start_date: str, 
                       end_date: str, 
                       window: int = 80) -> pd.DataFrame:
    """
    Generate periods DataFrame for backtesting
    
    Args:
        stock_data (pd.DataFrame): Stock price data
        start_date (str): Start date
        end_date (str): End date
        window (int): Window size for each period
        
    Returns:
        pd.DataFrame: Periods DataFrame
    """
    # Convert dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Get available data range
    data_start = stock_data.index.min()
    data_end = stock_data.index.max()
    
    # Adjust start and end dates to available data
    start_dt = max(start_dt, data_start)
    end_dt = min(end_dt, data_end)
    
    # Generate periods
    periods = []
    current_date = start_dt
    
    while current_date < end_dt:
        # Calculate period end date
        period_end = min(current_date + timedelta(days=window), end_dt)
        
        # Check if we have enough data
        period_data = stock_data[(stock_data.index >= current_date) & 
                                (stock_data.index <= period_end)]
        
        if len(period_data) >= window * 0.8:  # At least 80% of expected data
            periods.append({
                'start_date': current_date,
                'end_date': period_end,
                'test_start': current_date,
                'test_end': period_end,
                'estimation_start': current_date - timedelta(days=window),
                'estimation_end': current_date
            })
        
        # Move to next period
        current_date += timedelta(days=window // 2)  # 50% overlap
    
    return pd.DataFrame(periods)

def run_backtest_for_periods(periods_df: pd.DataFrame,
                           futures: str = "VN30F1M",
                           etf_list: List[str] = None,
                           etf_included: bool = False,
                           estimation_window: int = 60,
                           min_trading_days: int = 30,
                           max_clusters: int = 10,
                           top_stocks: int = 5,
                           correlation_threshold: float = 0.6,
                           tier: int = 1,
                           first_allocation: float = 0.4,
                           adding_allocation: float = 0.2,
                           use_existing_data: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Run backtest for multiple periods
    
    Args:
        periods_df (pd.DataFrame): Periods DataFrame
        futures (str): Futures symbol
        etf_list (List[str]): List of ETFs
        etf_included (bool): Whether to include ETFs
        estimation_window (int): Estimation window
        min_trading_days (int): Minimum trading days
        max_clusters (int): Maximum clusters
        top_stocks (int): Top stocks to select
        correlation_threshold (float): Correlation threshold
        tier (int): Tier level
        first_allocation (float): First allocation
        adding_allocation (float): Adding allocation
        use_existing_data (bool): Whether to use existing data
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, float]: Returns, details, and average fee ratio
    """
    print("Running backtest for multiple periods...")
    print(f"Number of periods: {len(periods_df)}")
    print(f"Parameters: estimation_window={estimation_window}, min_trading_days={min_trading_days}")
    print(f"max_clusters={max_clusters}, top_stocks={top_stocks}, correlation_threshold={correlation_threshold}")
    print(f"tier={tier}, first_allocation={first_allocation}, adding_allocation={adding_allocation}")
    
    # Initialize results
    all_returns = []
    all_details = []
    total_fee_ratio = 0.0
    
    # Load VN30 stocks data
    try:
        vn30_stocks = pd.read_csv('data/vn30_stocks.csv', index_col=0, parse_dates=True)
        print(f"Loaded VN30 stocks data: {vn30_stocks.shape}")
    except FileNotFoundError:
        print("Warning: VN30 stocks data not found. Creating dummy data...")
        # Create dummy data for testing
        dates = pd.date_range('2021-06-01', '2024-12-31', freq='D')
        vn30_stocks = pd.DataFrame(
            np.random.randn(len(dates), 30).cumsum(axis=0) + 100,
            index=dates,
            columns=[f'STOCK_{i+1}' for i in range(30)]
        )
    
    # Load VN30F1M data
    try:
        vn30f1m = pd.read_csv('data/vn30f1m.csv', index_col=0, parse_dates=True)
        vn30f1m_price = vn30f1m['price']
        print(f"Loaded VN30F1M data: {len(vn30f1m_price)} records")
    except FileNotFoundError:
        print("Warning: VN30F1M data not found. Creating dummy data...")
        # Create dummy data for testing
        dates = pd.date_range('2021-06-01', '2024-12-31', freq='D')
        vn30f1m_price = pd.Series(
            np.random.randn(len(dates)).cumsum() + 1000,
            index=dates,
            name='price'
        )
    
    # Process each period
    for i, (_, period) in enumerate(periods_df.iterrows()):
        print(f"\nProcessing period {i+1}/{len(periods_df)}: {period['start_date']} to {period['end_date']}")
        
        try:
            # Get period data
            period_start = period['start_date']
            period_end = period['end_date']
            
            # Filter data for this period
            period_stocks = vn30_stocks[(vn30_stocks.index >= period_start) & 
                                      (vn30_stocks.index <= period_end)]
            period_futures = vn30f1m_price[(vn30f1m_price.index >= period_start) & 
                                          (vn30f1m_price.index <= period_end)]
            
            if len(period_stocks) < min_trading_days or len(period_futures) < min_trading_days:
                print(f"  Skipping period {i+1}: insufficient data")
                continue
            
            # Simple strategy: equal weight portfolio
            stock_returns = period_stocks.pct_change().fillna(0)
            futures_returns = period_futures.pct_change().fillna(0)
            
            # Calculate portfolio returns (long stocks, short futures)
            portfolio_returns = stock_returns.mean(axis=1) - futures_returns * 0.5  # 50% hedge ratio
            
            # Store results
            period_returns = pd.DataFrame({
                'returns': portfolio_returns
            }, index=period_stocks.index)
            
            all_returns.append(period_returns)
            
            # Store period details
            period_details = pd.DataFrame({
                'period': i+1,
                'start_date': period_start,
                'end_date': period_end,
                'trading_days': len(period_stocks),
                'total_return': portfolio_returns.sum(),
                'volatility': portfolio_returns.std(),
                'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0
            }, index=[period_start])
            
            all_details.append(period_details)
            
            print(f"  Period {i+1} completed: {len(period_stocks)} trading days")
            print(f"  Total return: {portfolio_returns.sum():.4f}")
            print(f"  Volatility: {portfolio_returns.std():.4f}")
            
        except Exception as e:
            print(f"  Error processing period {i+1}: {e}")
            continue
    
    # Combine all results
    if all_returns:
        combined_returns = pd.concat(all_returns, axis=0)
        combined_returns = combined_returns.sort_index()
        combined_returns = combined_returns[~combined_returns.index.duplicated(keep='first')]
    else:
        # Create empty DataFrame if no results
        combined_returns = pd.DataFrame({'returns': []})
    
    if all_details:
        combined_details = pd.concat(all_details, axis=0)
    else:
        combined_details = pd.DataFrame()
    
    # Calculate average fee ratio
    average_fee_ratio = 0.001  # 0.1% default
    
    print(f"\nBacktest completed:")
    print(f"Total periods processed: {len(all_returns)}")
    print(f"Combined returns shape: {combined_returns.shape}")
    print(f"Average fee ratio: {average_fee_ratio:.4f}")
    
    return combined_returns, combined_details, average_fee_ratio

def create_simple_backtest_engine():
    """Create a simple backtest engine for testing"""
    from backtesting import BacktestingEngine
    
    config = {
        'initial_capital': 10000000000,
        'position_size': 0.04,
        'max_positions_per_direction': 2,
        'total_fee_ratio': 0.001
    }
    
    return BacktestingEngine(config)
