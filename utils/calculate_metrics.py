"""
Metrics calculation functions for statistical arbitrage strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import os
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
        # Set matplotlib backend to avoid display issues
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Strategy Performance Analysis', fontsize=16)
        
        returns = returns_df['returns'].dropna()
        
        # 1. Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, linewidth=2)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Returns distribution
        axes[1, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(window=30).mean() / returns.rolling(window=30).std() * np.sqrt(252)
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='green')
        axes[1, 1].set_title('Rolling Sharpe Ratio (30-day)')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot with better error handling
        import os
        os.makedirs('result/plots', exist_ok=True)
        
        try:
            plt.savefig('result/plots/performance_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
            print("✅ Performance plots saved to result/plots/performance_analysis.png")
        except Exception as save_error:
            print(f"❌ Could not save plots to file: {save_error}")
        
        # Close the figure to free memory
        plt.close(fig)
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")

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

def integrate_trading_analysis():
    """
    Integrate existing trading signals, daily balance, and money changes into backtesting
    """
    print('\n' + '=' * 80)
    print('INTEGRATING TRADING ANALYSIS INTO BACKTESTING')
    print('=' * 80)
    
    try:
        # Check if any trading analysis files exist
        trading_files = [
            'result/enhanced_trading_signals.csv',
            'result/enhanced_daily_balance.csv', 
            'result/enhanced_asset_summary.csv',
            'result/enhanced_performance_summary.csv',
            'result/all_trading_signals.csv',
            'result/money_changes_trading_signals.csv',
            'result/daily_balance_summary.csv'
        ]
        
        existing_files = []
        for file in trading_files:
            if os.path.exists(file):
                existing_files.append(file)
                print(f'✅ Found: {file}')
            else:
                print(f'❌ Missing: {file}')
        
        if not existing_files:
            print('⚠️  No trading analysis files found. Running basic analysis...')
            return
        
        # Load and display trading signals summary
        if os.path.exists('result/enhanced_trading_signals.csv'):
            trades_df = pd.read_csv('result/enhanced_trading_signals.csv')
            print(f'\n📊 TRADING SIGNALS SUMMARY:')
            print(f'   Total Signals: {len(trades_df)}')
            print(f'   Assets: {trades_df["Asset"].nunique()}')
            print(f'   Signal Types: {trades_df["Signal_Type"].value_counts().to_dict()}')
            
            # Show recent signals
            recent_signals = trades_df.tail(5)[['Timestamp', 'Asset', 'Signal_Type', 'Trade_Profit_Loss', 'Cumulative_Balance']]
            print(f'\n📈 Recent Trading Signals:')
            print(recent_signals.to_string(index=False))
        
        # Load and display daily balance summary
        if os.path.exists('result/enhanced_daily_balance.csv'):
            daily_balance = pd.read_csv('result/enhanced_daily_balance.csv')
            print(f'\n💰 DAILY BALANCE SUMMARY:')
            print(f'   Initial Balance: ${daily_balance["Cumulative_Balance"].iloc[0]:,.2f}')
            print(f'   Final Balance: ${daily_balance["Cumulative_Balance"].iloc[-1]:,.2f}')
            print(f'   Total Change: ${daily_balance["Cumulative_Balance"].iloc[-1] - daily_balance["Cumulative_Balance"].iloc[0]:,.2f}')
            print(f'   Trading Days: {len(daily_balance)}')
            
            # Show recent daily changes
            recent_daily = daily_balance.tail(5)[['Timestamp', 'Cumulative_Balance', 'Daily_Change', 'Daily_Return']]
            print(f'\n📅 Recent Daily Balance:')
            print(recent_daily.to_string(index=False))
        
        # Load and display asset summary
        if os.path.exists('result/enhanced_asset_summary.csv'):
            asset_summary = pd.read_csv('result/enhanced_asset_summary.csv', index_col=0)
            print(f'\n🏆 ASSET PERFORMANCE SUMMARY:')
            for asset in asset_summary.index:
                pnl = asset_summary.loc[asset, 'Trade_Profit_Loss_sum']
                return_pct = asset_summary.loc[asset, 'Total_Return']
                print(f'   {asset}: P&L ${pnl:,.2f} ({return_pct:.2f}%)')
        
        # Load and display performance summary
        if os.path.exists('result/enhanced_performance_summary.csv'):
            performance = pd.read_csv('result/enhanced_performance_summary.csv')
            print(f'\n🎯 OVERALL PERFORMANCE:')
            print(f'   Initial Capital: ${performance["Initial_Capital"].iloc[0]:,.2f}')
            print(f'   Final Balance: ${performance["Final_Balance"].iloc[0]:,.2f}')
            print(f'   Total Return: {performance["Total_Return"].iloc[0]:.2f}%')
            print(f'   Total Trades: {performance["Total_Trades"].iloc[0]}')
            print(f'   Win Rate: {performance["Win_Rate"].iloc[0]:.2f}%')
            print(f'   Best Trade: ${performance["Best_Trade"].iloc[0]:.2f}')
            print(f'   Worst Trade: ${performance["Worst_Trade"].iloc[0]:.2f}')
        
        # Create integrated visualization
        create_integrated_visualization()
        
        print(f'\n✅ Trading analysis integration completed!')
        print(f'   All trading signals, daily balance, and money changes are now integrated into backtesting.')
        
    except Exception as e:
        print(f'❌ Error integrating trading analysis: {e}')

def create_integrated_visualization():
    """
    Create integrated visualization combining all trading analysis
    """
    try:
        # Set matplotlib backend
        import matplotlib
        matplotlib.use('Agg')
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Integrated Trading Analysis - All Signals, Balance & Money Changes', fontsize=16, fontweight='bold')
        
        # Plot 1: Trading signals over time
        if os.path.exists('result/enhanced_trading_signals.csv'):
            trades_df = pd.read_csv('result/enhanced_trading_signals.csv')
            trades_df['Timestamp'] = pd.to_datetime(trades_df['Timestamp'])
            
            # Count signals by date
            daily_signals = trades_df.groupby(trades_df['Timestamp'].dt.date).size()
            
            ax1 = axes[0, 0]
            ax1.plot(daily_signals.index, daily_signals.values, linewidth=2, color='blue')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Daily Signals')
            ax1.set_title('Trading Signals Over Time')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative balance
        if os.path.exists('result/enhanced_daily_balance.csv'):
            daily_balance = pd.read_csv('result/enhanced_daily_balance.csv')
            daily_balance['Timestamp'] = pd.to_datetime(daily_balance['Timestamp'])
            
            ax2 = axes[0, 1]
            ax2.plot(daily_balance['Timestamp'], daily_balance['Cumulative_Balance'], linewidth=2, color='green')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative Balance ($)')
            ax2.set_title('Cumulative Balance Over Time')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Daily balance changes
        if os.path.exists('result/enhanced_daily_balance.csv'):
            ax3 = axes[0, 2]
            ax3.bar(daily_balance['Timestamp'], daily_balance['Daily_Change'], alpha=0.7, color='orange')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Daily Balance Change ($)')
            ax3.set_title('Daily Balance Changes')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Asset performance
        if os.path.exists('result/enhanced_asset_summary.csv'):
            asset_summary = pd.read_csv('result/enhanced_asset_summary.csv', index_col=0)
            
            ax4 = axes[1, 0]
            assets = asset_summary.index.tolist()
            pnl_values = asset_summary['Trade_Profit_Loss_sum'].tolist()
            colors = ['green' if x > 0 else 'red' for x in pnl_values]
            
            ax4.bar(assets, pnl_values, color=colors, alpha=0.7)
            ax4.set_xlabel('Assets')
            ax4.set_ylabel('Total P&L ($)')
            ax4.set_title('Asset Performance')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Signal type distribution
        if os.path.exists('result/enhanced_trading_signals.csv'):
            ax5 = axes[1, 1]
            signal_counts = trades_df['Signal_Type'].value_counts()
            ax5.pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%', startangle=90)
            ax5.set_title('Signal Type Distribution')
        
        # Plot 6: Performance summary
        if os.path.exists('result/enhanced_performance_summary.csv'):
            performance = pd.read_csv('result/enhanced_performance_summary.csv')
            
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            # Create performance text
            perf_text = f"""
PERFORMANCE SUMMARY

Initial Capital: ${performance['Initial_Capital'].iloc[0]:,.2f}
Final Balance: ${performance['Final_Balance'].iloc[0]:,.2f}
Total Return: {performance['Total_Return'].iloc[0]:.2f}%

Total Trades: {performance['Total_Trades'].iloc[0]}
Win Rate: {performance['Win_Rate'].iloc[0]:.2f}%
Best Trade: ${performance['Best_Trade'].iloc[0]:.2f}
Worst Trade: ${performance['Worst_Trade'].iloc[0]:.2f}
            """
            
            ax6.text(0.1, 0.9, perf_text, transform=ax6.transAxes, fontsize=10, 
                    verticalalignment='top', fontfamily='monospace')
            ax6.set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save the integrated plot
        plt.savefig('result/plots/integrated_trading_analysis.png', dpi=300, bbox_inches='tight')
        print('✅ Integrated trading analysis plot saved to result/plots/integrated_trading_analysis.png')
        
        plt.close(fig)
        
    except Exception as e:
        print(f'❌ Error creating integrated visualization: {e}')

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
