"""
Metrics calculation functions for statistical arbitrage strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(returns_df: pd.DataFrame, 
                     average_fee_ratio: float = 0.001,
                     risk_free_rate: float = 0.05,
                     plotting: bool = True,
                     use_existing_data: bool = True) -> Dict:
    """
    Calculate performance metrics for the strategy
    
    Args:
        returns_df (pd.DataFrame): Strategy returns
        average_fee_ratio (float): Average fee ratio
        risk_free_rate (float): Risk-free rate
        plotting (bool): Whether to generate plots
        use_existing_data (bool): Whether to use existing data
        
    Returns:
        Dict: Performance metrics
    """
    if returns_df.empty or len(returns_df) == 0:
        print("Warning: No returns data available")
        return {'error': 'No data available'}
    
    # Calculate basic metrics
    returns = returns_df['returns'].dropna()
    
    if len(returns) == 0:
        print("Warning: No valid returns data")
        return {'error': 'No valid returns'}
    
    # Basic statistics
    total_return = returns.sum()
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Create results dictionary
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(returns),
        'avg_return': returns.mean(),
        'median_return': returns.median(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis()
    }
    
    # Print results
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    print(f"Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"Annual Return: {annual_return:.4f} ({annual_return*100:.2f}%)")
    print(f"Volatility: {volatility:.4f} ({volatility*100:.2f}%)")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
    print(f"Win Rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
    print(f"Profit Factor: {profit_factor:.4f}")
    print(f"Total Trades: {len(returns)}")
    print(f"Average Return: {returns.mean():.6f}")
    print(f"Median Return: {returns.median():.6f}")
    print(f"Skewness: {returns.skew():.4f}")
    print(f"Kurtosis: {returns.kurtosis():.4f}")
    print("="*60)
    
    # Generate plots if requested
    if plotting:
        try:
            generate_performance_plots(returns_df, metrics)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    return metrics

def generate_performance_plots(returns_df: pd.DataFrame, metrics: Dict):
    """Generate performance plots"""
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Strategy Performance Analysis', fontsize=16)
        
        returns = returns_df['returns'].dropna()
        
        # 1. Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True)
        
        # 2. Drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True)
        
        # 3. Returns distribution
        axes[1, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # 4. Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(window=30).mean() / returns.rolling(window=30).std() * np.sqrt(252)
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[1, 1].set_title('Rolling Sharpe Ratio (30-day)')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        try:
            plt.savefig('result/plots/performance_analysis.png', dpi=300, bbox_inches='tight')
            print("Performance plots saved to result/plots/performance_analysis.png")
        except:
            print("Could not save plots to file")
        
        plt.show()
        
    except Exception as e:
        print(f"Error generating plots: {e}")

def calculate_monthly_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly returns
    
    Args:
        returns_df (pd.DataFrame): Daily returns
        
    Returns:
        pd.DataFrame: Monthly returns
    """
    if returns_df.empty:
        return pd.DataFrame()
    
    returns = returns_df['returns'].dropna()
    
    # Resample to monthly
    monthly_returns = (1 + returns).resample('M').prod() - 1
    
    return monthly_returns

def pivot_monthly_returns_to_table(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly returns to pivot table format
    
    Args:
        monthly_returns (pd.DataFrame): Monthly returns
        
    Returns:
        pd.DataFrame: Pivot table
    """
    if monthly_returns.empty:
        return pd.DataFrame()
    
    # Create year and month columns
    monthly_returns_df = monthly_returns.to_frame('returns')
    monthly_returns_df['year'] = monthly_returns_df.index.year
    monthly_returns_df['month'] = monthly_returns_df.index.month
    
    # Create pivot table
    pivot_table = monthly_returns_df.pivot_table(
        values='returns', 
        index='year', 
        columns='month', 
        aggfunc='sum'
    )
    
    # Add annual totals
    pivot_table['Annual'] = pivot_table.sum(axis=1)
    
    # Format as percentage
    pivot_table = pivot_table * 100
    
    return pivot_table

def calculate_shapre_and_mdd(returns_df: pd.DataFrame, risk_free_rate: float = 0.05) -> Tuple[float, float, float]:
    """
    Calculate Sharpe ratio and Maximum Drawdown
    
    Args:
        returns_df (pd.DataFrame): Strategy returns
        risk_free_rate (float): Risk-free rate
        
    Returns:
        Tuple[float, float, float]: (annual_return, sharpe_ratio, max_drawdown)
    """
    if returns_df.empty:
        return 0.0, 0.0, 0.0
    
    returns = returns_df['returns'].dropna()
    
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    
    # Annual return
    total_return = returns.sum()
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    # Sharpe ratio
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return annual_return, sharpe_ratio, max_drawdown
