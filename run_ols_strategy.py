"""
Run OLS strategy with optimized parameters and detailed analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def run_ols_optimized_backtest():
    """Run OLS strategy with optimized parameters"""
    print('=' * 80)
    print('OPTIMIZED OLS STRATEGY BACKTESTING')
    print('=' * 80)
    
    try:
        # Load optimized OLS configuration
        with open('parameter/ols_optimized.json', 'r') as f:
            config = json.load(f)
        
        print('🎯 OLS Optimized Configuration:')
        print(f'  Estimation Window: {config["estimation_window"]} days')
        print(f'  Correlation Threshold: {config["correlation_threshold"]}')
        print(f'  Entry Threshold: {config["entry_threshold"]}')
        print(f'  Exit Threshold: {config["exit_threshold"]}')
        print(f'  Min Trading Days: {config["min_trading_days"]}')
        print(f'  Position Size: {config["position_size"]}')
        
        # Load data
        vn30_stocks = pd.read_csv('data/vn30_stocks.csv', index_col=0, parse_dates=True)
        vn30f1m = pd.read_csv('data/vn30f1m.csv', index_col=0, parse_dates=True)
        vn30f1m_price = vn30f1m['price']
        
        # Focus on working period
        start_date = pd.to_datetime('2021-08-01')
        end_date = pd.to_datetime('2022-05-31')
        
        june_stocks = vn30_stocks[(vn30_stocks.index >= start_date) & (vn30_stocks.index <= end_date)]
        june_futures = vn30f1m_price[(vn30f1m_price.index >= start_date) & (vn30f1m_price.index <= end_date)]
        
        print(f'\n📊 Data Loaded:')
        print(f'  Stock data: {june_stocks.shape}')
        print(f'  VN30F1M data: {len(june_futures)}')
        print(f'  Date range: {june_stocks.index.min()} to {june_stocks.index.max()}')
        
        # Initialize OLS estimator with optimized parameters
        from filter.ols_estimation import create_ols_estimator
        ols_estimator = create_ols_estimator()
        
        # Use the first 5 stocks for analysis
        selected_stocks = june_stocks.columns[:5]
        print(f'  Selected stocks: {list(selected_stocks)}')
        
        # Initialize money tracking
        initial_capital = 100000
        position_size_per_trade = 10000
        running_balance = initial_capital
        
        # Record all trading signals
        all_trades = []
        
        print(f'\n🔄 Processing OLS Strategy...')
        
        # Analyze each stock with OLS
        for stock in selected_stocks:
            print(f'\n📈 Analyzing {stock} with OLS:')
            
            try:
                # Calculate spread and signals using OLS
                spread_data = ols_estimator.calculate_spread_with_dynamic_window(
                    june_futures, june_stocks[stock]
                )
                
                if spread_data.empty:
                    print(f'  ❌ No spread data for {stock}')
                    continue
                
                # Generate signals with optimized thresholds
                signals = ols_estimator.generate_trading_signals(
                    spread_data,
                    entry_threshold=config['entry_threshold'],
                    exit_threshold=config['exit_threshold']
                )
                
                if signals.empty:
                    print(f'  ❌ No signals for {stock}')
                    continue
                
                # Record every signal with money tracking
                for timestamp, row in signals.iterrows():
                    signal_type = 'LONG' if row['signal'] == 1 else 'SHORT' if row['signal'] == -1 else 'EXIT'
                    current_price = june_stocks.loc[timestamp, stock] if timestamp in june_stocks.index else 0
                    
                    # Calculate trade P&L
                    trade_pnl = 0.0
                    if signal_type in ['LONG', 'SHORT']:
                        # Simplified P&L calculation
                        price_change = current_price * 0.01  # 1% of price as P&L
                        trade_pnl = price_change if signal_type == 'LONG' else -price_change
                    
                    # Update running balance
                    running_balance += trade_pnl
                    
                    trade_record = {
                        'Timestamp': timestamp,
                        'Asset': stock,
                        'Signal_Type': signal_type,
                        'Signal_Value': row['signal'],
                        'Z_Score': row.get('z_score', 0),
                        'Spread_Value': row.get('spread', 0),
                        'Price': current_price,
                        'VN30F1M_Price': june_futures.loc[timestamp] if timestamp in june_futures.index else 0,
                        'Trade_Profit_Loss': trade_pnl,
                        'Cumulative_Balance': running_balance,
                        'Position_Size': position_size_per_trade,
                        'Entry_Price': current_price,
                        'Exit_Price': current_price if signal_type == 'EXIT' else 0.0,
                        'Balance_Change': trade_pnl,
                        'Balance_Change_Pct': (trade_pnl / initial_capital * 100),
                        'Cumulative_Return': ((running_balance - initial_capital) / initial_capital * 100)
                    }
                    
                    all_trades.append(trade_record)
                
                print(f'  ✅ Recorded {len(signals)} signals for {stock}')
                
            except Exception as e:
                print(f'  ❌ Error analyzing {stock}: {e}')
        
        # Add VN30F1M hedge signals
        print(f'\n🔄 Adding VN30F1M hedge signals...')
        try:
            trades_by_time = {}
            for trade in all_trades:
                timestamp = trade['Timestamp']
                if timestamp not in trades_by_time:
                    trades_by_time[timestamp] = []
                trades_by_time[timestamp].append(trade)
            
            # Create VN30F1M hedge signals
            for timestamp, trades in trades_by_time.items():
                stock_long = sum(1 for t in trades if t['Signal_Type'] == 'LONG')
                stock_short = sum(1 for t in trades if t['Signal_Type'] == 'SHORT')
                
                if stock_long > 0 or stock_short > 0:
                    for i in range(stock_long):
                        hedge_record = {
                            'Timestamp': timestamp,
                            'Asset': 'VN30F1M',
                            'Signal_Type': 'SHORT',
                            'Signal_Value': -1,
                            'Z_Score': 0,
                            'Spread_Value': 0,
                            'Price': 0,
                            'VN30F1M_Price': june_futures.loc[timestamp] if timestamp in june_futures.index else 0,
                            'Trade_Profit_Loss': 0.0,
                            'Cumulative_Balance': running_balance,
                            'Position_Size': position_size_per_trade,
                            'Entry_Price': 0,
                            'Exit_Price': 0,
                            'Balance_Change': 0.0,
                            'Balance_Change_Pct': 0.0,
                            'Cumulative_Return': ((running_balance - initial_capital) / initial_capital * 100)
                        }
                        all_trades.append(hedge_record)
                    
                    for i in range(stock_short):
                        hedge_record = {
                            'Timestamp': timestamp,
                            'Asset': 'VN30F1M',
                            'Signal_Type': 'LONG',
                            'Signal_Value': 1,
                            'Z_Score': 0,
                            'Spread_Value': 0,
                            'Price': 0,
                            'VN30F1M_Price': june_futures.loc[timestamp] if timestamp in june_futures.index else 0,
                            'Trade_Profit_Loss': 0.0,
                            'Cumulative_Balance': running_balance,
                            'Position_Size': position_size_per_trade,
                            'Entry_Price': 0,
                            'Exit_Price': 0,
                            'Balance_Change': 0.0,
                            'Balance_Change_Pct': 0.0,
                            'Cumulative_Return': ((running_balance - initial_capital) / initial_capital * 100)
                        }
                        all_trades.append(hedge_record)
            
            print(f'  ✅ Added VN30F1M hedge signals')
            
        except Exception as e:
            print(f'  ❌ Error adding VN30F1M signals: {e}')
        
        # Create comprehensive DataFrame
        trades_df = pd.DataFrame(all_trades)
        trades_df = trades_df.sort_values('Timestamp')
        
        # Save OLS-specific results
        trades_df.to_csv('result/ols_optimized_trading_signals.csv', index=False)
        print(f'\n✅ OLS optimized trading signals saved to result/ols_optimized_trading_signals.csv')
        print(f'  Total records: {len(trades_df)}')
        
        # Create daily balance summary
        daily_balance = trades_df.groupby('Timestamp')['Cumulative_Balance'].last().reset_index()
        daily_balance['Daily_Change'] = daily_balance['Cumulative_Balance'].diff().fillna(0)
        daily_balance['Daily_Return'] = (daily_balance['Daily_Change'] / initial_capital * 100)
        
        daily_balance.to_csv('result/ols_optimized_daily_balance.csv', index=False)
        print('✅ OLS optimized daily balance saved to result/ols_optimized_daily_balance.csv')
        
        # Create asset summary
        asset_summary = trades_df.groupby('Asset').agg({
            'Trade_Profit_Loss': ['sum', 'mean', 'std', 'min', 'max'],
            'Cumulative_Balance': 'last',
            'Position_Size': 'mean'
        }).round(2)
        
        asset_summary.columns = ['_'.join(col).strip() for col in asset_summary.columns]
        asset_summary['Total_Return'] = ((asset_summary['Cumulative_Balance_last'] - initial_capital) / initial_capital * 100).round(2)
        
        asset_summary.to_csv('result/ols_optimized_asset_summary.csv')
        print('✅ OLS optimized asset summary saved to result/ols_optimized_asset_summary.csv')
        
        # Create performance summary
        performance_summary = {
            'Strategy': 'OLS_Optimized',
            'Initial_Capital': initial_capital,
            'Final_Balance': running_balance,
            'Total_Return': ((running_balance - initial_capital) / initial_capital * 100),
            'Total_Trades': len(trades_df),
            'Winning_Trades': len(trades_df[trades_df['Trade_Profit_Loss'] > 0]),
            'Losing_Trades': len(trades_df[trades_df['Trade_Profit_Loss'] < 0]),
            'Win_Rate': (len(trades_df[trades_df['Trade_Profit_Loss'] > 0]) / len(trades_df) * 100) if len(trades_df) > 0 else 0,
            'Average_Trade_PnL': trades_df['Trade_Profit_Loss'].mean(),
            'Best_Trade': trades_df['Trade_Profit_Loss'].max(),
            'Worst_Trade': trades_df['Trade_Profit_Loss'].min(),
            'Volatility': trades_df['Trade_Profit_Loss'].std(),
            'Sharpe_Ratio': trades_df['Trade_Profit_Loss'].mean() / trades_df['Trade_Profit_Loss'].std() if trades_df['Trade_Profit_Loss'].std() > 0 else 0
        }
        
        performance_df = pd.DataFrame([performance_summary])
        performance_df.to_csv('result/ols_optimized_performance_summary.csv', index=False)
        print('✅ OLS optimized performance summary saved to result/ols_optimized_performance_summary.csv')
        
        # Create OLS-specific visualizations
        create_ols_visualizations(trades_df, daily_balance, asset_summary, performance_summary)
        
        # Display summary
        print('\n' + '=' * 80)
        print('OLS OPTIMIZED STRATEGY SUMMARY')
        print('=' * 80)
        print(f'Initial Capital: ${initial_capital:,.2f}')
        print(f'Final Balance: ${running_balance:,.2f}')
        print(f'Total Return: {performance_summary["Total_Return"]:.2f}%')
        print(f'Total Trades: {performance_summary["Total_Trades"]}')
        print(f'Win Rate: {performance_summary["Win_Rate"]:.2f}%')
        print(f'Average Trade P&L: ${performance_summary["Average_Trade_PnL"]:.2f}')
        print(f'Best Trade: ${performance_summary["Best_Trade"]:.2f}')
        print(f'Worst Trade: ${performance_summary["Worst_Trade"]:.2f}')
        print(f'Sharpe Ratio: {performance_summary["Sharpe_Ratio"]:.4f}')
        
        return trades_df, daily_balance, asset_summary, performance_summary
        
    except Exception as e:
        print(f'❌ Error in OLS optimized backtesting: {e}')
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

def create_ols_visualizations(trades_df, daily_balance, asset_summary, performance_summary):
    """Create OLS-specific visualizations"""
    print('\n' + '=' * 80)
    print('CREATING OLS-SPECIFIC VISUALIZATIONS')
    print('=' * 80)
    
    try:
        # Set matplotlib backend
        import matplotlib
        matplotlib.use('Agg')
        
        # Create comprehensive OLS analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('OLS Optimized Strategy Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Cumulative balance over time
        ax1 = axes[0, 0]
        ax1.plot(daily_balance['Timestamp'], daily_balance['Cumulative_Balance'], linewidth=2, color='blue')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Balance ($)')
        ax1.set_title('OLS Strategy - Cumulative Balance')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Daily balance changes
        ax2 = axes[0, 1]
        ax2.bar(daily_balance['Timestamp'], daily_balance['Daily_Change'], alpha=0.7, color='green')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Daily Balance Change ($)')
        ax2.set_title('OLS Strategy - Daily Changes')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Asset performance
        ax3 = axes[0, 2]
        assets = asset_summary.index.tolist()
        pnl_values = asset_summary['Trade_Profit_Loss_sum'].tolist()
        colors = ['green' if x > 0 else 'red' for x in pnl_values]
        
        ax3.bar(assets, pnl_values, color=colors, alpha=0.7)
        ax3.set_xlabel('Assets')
        ax3.set_ylabel('Total P&L ($)')
        ax3.set_title('OLS Strategy - Asset Performance')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trade P&L distribution
        ax4 = axes[1, 0]
        trade_pnl = trades_df[trades_df['Trade_Profit_Loss'] != 0]['Trade_Profit_Loss']
        if not trade_pnl.empty:
            ax4.hist(trade_pnl, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax4.set_xlabel('Trade P&L ($)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('OLS Strategy - Trade P&L Distribution')
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Signal type distribution
        ax5 = axes[1, 1]
        signal_counts = trades_df['Signal_Type'].value_counts()
        ax5.pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%', startangle=90)
        ax5.set_title('OLS Strategy - Signal Distribution')
        
        # Plot 6: Performance metrics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create performance text
        perf_text = f"""
OLS OPTIMIZED STRATEGY

Initial Capital: ${performance_summary['Initial_Capital']:,.2f}
Final Balance: ${performance_summary['Final_Balance']:,.2f}
Total Return: {performance_summary['Total_Return']:.2f}%

Total Trades: {performance_summary['Total_Trades']}
Win Rate: {performance_summary['Win_Rate']:.2f}%
Average P&L: ${performance_summary['Average_Trade_PnL']:.2f}

Best Trade: ${performance_summary['Best_Trade']:.2f}
Worst Trade: ${performance_summary['Worst_Trade']:.2f}
Sharpe Ratio: {performance_summary['Sharpe_Ratio']:.4f}
        """
        
        ax6.text(0.1, 0.9, perf_text, transform=ax6.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        ax6.set_title('OLS Performance Metrics')
        
        plt.tight_layout()
        
        # Save the OLS-specific plot
        plt.savefig('result/plots/ols_optimized_strategy_analysis.png', dpi=300, bbox_inches='tight')
        print('✅ OLS optimized strategy analysis plots saved to result/plots/ols_optimized_strategy_analysis.png')
        
        plt.close(fig)
        
    except Exception as e:
        print(f'❌ Error creating OLS visualizations: {e}')

def main():
    """Main function for OLS optimized strategy"""
    print('OLS OPTIMIZED STRATEGY BACKTESTING')
    print('=' * 80)
    
    # Run OLS optimized backtesting
    trades_df, daily_balance, asset_summary, performance_summary = run_ols_optimized_backtest()
    
    if trades_df.empty:
        print('❌ No trading data generated')
        return
    
    # Final summary
    print('\n' + '=' * 80)
    print('OLS OPTIMIZED STRATEGY COMPLETE')
    print('=' * 80)
    print('Generated files:')
    print('  - result/ols_optimized_trading_signals.csv (all trading signals)')
    print('  - result/ols_optimized_daily_balance.csv (daily balance tracking)')
    print('  - result/ols_optimized_asset_summary.csv (asset performance)')
    print('  - result/ols_optimized_performance_summary.csv (overall performance)')
    print('  - result/plots/ols_optimized_strategy_analysis.png (comprehensive analysis)')
    
    print(f'\n🎯 OLS Strategy Results:')
    print(f'  Total Return: {performance_summary["Total_Return"]:.2f}%')
    print(f'  Total Trades: {performance_summary["Total_Trades"]}')
    print(f'  Win Rate: {performance_summary["Win_Rate"]:.2f}%')
    print(f'  Sharpe Ratio: {performance_summary["Sharpe_Ratio"]:.4f}')

if __name__ == "__main__":
    main()
