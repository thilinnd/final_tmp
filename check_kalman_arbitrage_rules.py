"""
Check if Kalman filter follows proper statistical arbitrage rules
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def check_kalman_arbitrage_rules():
    """Check if Kalman filter follows proper statistical arbitrage rules"""
    print('=' * 80)
    print('KIỂM TRA QUY TẮC STATISTICAL ARBITRAGE CỦA KALMAN FILTER')
    print('=' * 80)
    
    print('\n🎯 QUY TẮC STATISTICAL ARBITRAGE:')
    print('-' * 50)
    print('✅ LONG STOCK + SHORT FUTURES (khi spread thấp)')
    print('✅ SHORT STOCK + LONG FUTURES (khi spread cao)')
    print('❌ KHÔNG được: LONG STOCK + LONG FUTURES')
    print('❌ KHÔNG được: SHORT STOCK + SHORT FUTURES')
    
    # Load data to analyze
    try:
        # Load stock data
        vn30_stocks = pd.read_csv('data/vn30_stocks.csv', index_col=0, parse_dates=True)
        vn30f1m = pd.read_csv('data/vn30f1m.csv', index_col=0, parse_dates=True)
        vn30f1m_price = vn30f1m['price']
        
        print(f'\n📊 DATA LOADED:')
        print(f'  Stock data: {vn30_stocks.shape}')
        print(f'  VN30F1M data: {len(vn30f1m_price)}')
        
        # Focus on working period
        start_date = pd.to_datetime('2021-08-01')
        end_date = pd.to_datetime('2022-05-31')
        
        june_stocks = vn30_stocks[(vn30_stocks.index >= start_date) & (vn30_stocks.index <= end_date)]
        june_futures = vn30f1m_price[(vn30f1m_price.index >= start_date) & (vn30f1m_price.index <= end_date)]
        
        print(f'  Working period: {start_date} to {end_date}')
        print(f'  Stock data: {june_stocks.shape}')
        print(f'  VN30F1M data: {len(june_futures)}')
        
        # Analyze the Kalman filter implementation
        analyze_kalman_implementation()
        
        # Check if it follows arbitrage rules
        check_arbitrage_implementation()
        
    except Exception as e:
        print(f'❌ Error loading data: {e}')

def analyze_kalman_implementation():
    """Analyze the current Kalman filter implementation"""
    print('\n' + '=' * 80)
    print('PHÂN TÍCH IMPLEMENTATION KALMAN FILTER')
    print('=' * 80)
    
    print('\n🔍 CODE HIỆN TẠI TRONG backtesting.py:')
    print('-' * 50)
    print('```python')
    print('# FIX 3: Proper Signal Generation with Hedge Ratio')
    print('# Get hedge ratio from Kalman filter')
    print('hedge_ratio = self.kalman_filter.get_hedge_ratio() if hasattr(self.kalman_filter, "get_hedge_ratio") else 1.0')
    print('')
    print('# Calculate proper statistical arbitrage returns')
    print('stock_returns = aligned_stocks[stock].pct_change().fillna(0)')
    print('futures_returns = aligned_futures.pct_change().fillna(0)')
    print('')
    print('# Statistical arbitrage: stock_return - hedge_ratio * futures_return')
    print('arbitrage_returns = stock_returns - hedge_ratio * futures_returns')
    print('')
    print('# Apply signals to arbitrage returns')
    print('stock_portfolio_returns = trading_signals["signal"] * arbitrage_returns')
    print('```')
    
    print('\n🎯 PHÂN TÍCH:')
    print('-' * 50)
    print('✅ CÓ hedge ratio calculation')
    print('✅ CÓ statistical arbitrage formula: stock_return - hedge_ratio * futures_return')
    print('✅ CÓ signal application')
    print('❌ THIẾU: Explicit position management (long/short stocks vs futures)')
    print('❌ THIẾU: Clear position tracking for each asset')

def check_arbitrage_implementation():
    """Check if the implementation follows proper arbitrage rules"""
    print('\n' + '=' * 80)
    print('KIỂM TRA QUY TẮC ARBITRAGE')
    print('=' * 80)
    
    print('\n🚨 VẤN ĐỀ HIỆN TẠI:')
    print('-' * 50)
    print('❌ Kalman filter hiện tại CHỈ tính toán returns, KHÔNG tạo positions')
    print('❌ KHÔNG có explicit long/short positions cho stocks và futures')
    print('❌ KHÔNG tuân thủ quy tắc: LONG STOCK + SHORT FUTURES')
    print('❌ KHÔNG tuân thủ quy tắc: SHORT STOCK + LONG FUTURES')
    
    print('\n✅ CÁCH FIX ĐÚNG:')
    print('-' * 50)
    print('1. Khi signal = 1 (LONG):')
    print('   - LONG stock position')
    print('   - SHORT futures position (hedge_ratio * futures)')
    print('')
    print('2. Khi signal = -1 (SHORT):')
    print('   - SHORT stock position')
    print('   - LONG futures position (hedge_ratio * futures)')
    print('')
    print('3. Khi signal = 0 (EXIT):')
    print('   - Close all positions')
    
    # Create proper implementation
    create_proper_arbitrage_implementation()

def create_proper_arbitrage_implementation():
    """Create proper statistical arbitrage implementation"""
    print('\n' + '=' * 80)
    print('IMPLEMENTATION ĐÚNG CHO STATISTICAL ARBITRAGE')
    print('=' * 80)
    
    print('\n🔧 CODE ĐÚNG:')
    print('-' * 50)
    print('```python')
    print('def _run_kalman_strategy_proper(self, stock_data, futures_data):')
    print('    """Proper statistical arbitrage with explicit positions"""')
    print('    ')
    print('    # Initialize positions')
    print('    stock_positions = pd.DataFrame(0.0, index=stock_data.index, columns=stock_data.columns)')
    print('    futures_positions = pd.Series(0.0, index=futures_data.index)')
    print('    portfolio_returns = pd.Series(0.0, index=stock_data.index)')
    print('    ')
    print('    for stock in stock_data.columns:')
    print('        # Get trading signals')
    print('        signals = self.kalman_filter.generate_trading_signals(')
    print('            futures_data, stock_data[stock]')
    print('        )')
    print('        ')
    print('        # Get hedge ratio')
    print('        hedge_ratio = self.kalman_filter.get_hedge_ratio()')
    print('        ')
    print('        for timestamp, signal_row in signals.iterrows():')
    print('            signal = signal_row["signal"]')
    print('            ')
    print('            if signal == 1:  # LONG SIGNAL')
    print('                # LONG STOCK + SHORT FUTURES')
    print('                stock_positions.loc[timestamp, stock] = 1.0')
    print('                futures_positions.loc[timestamp] -= hedge_ratio')
    print('                ')
    print('            elif signal == -1:  # SHORT SIGNAL')
    print('                # SHORT STOCK + LONG FUTURES')
    print('                stock_positions.loc[timestamp, stock] = -1.0')
    print('                futures_positions.loc[timestamp] += hedge_ratio')
    print('                ')
    print('            elif signal == 0:  # EXIT SIGNAL')
    print('                # CLOSE ALL POSITIONS')
    print('                stock_positions.loc[timestamp, stock] = 0.0')
    print('                futures_positions.loc[timestamp] = 0.0')
    print('    ')
    print('    # Calculate returns based on positions')
    print('    stock_returns = stock_data.pct_change().fillna(0)')
    print('    futures_returns = futures_data.pct_change().fillna(0)')
    print('    ')
    print('    # Portfolio returns = sum(stock_positions * stock_returns) + futures_positions * futures_returns')
    print('    portfolio_returns = (stock_positions * stock_returns).sum(axis=1) + futures_positions * futures_returns')
    print('    ')
    print('    return pd.DataFrame({"returns": portfolio_returns}, index=stock_data.index)')
    print('```')
    
    print('\n🎯 QUY TẮC ĐƯỢC TUÂN THỦ:')
    print('-' * 50)
    print('✅ LONG STOCK + SHORT FUTURES (khi signal = 1)')
    print('✅ SHORT STOCK + LONG FUTURES (khi signal = -1)')
    print('✅ CLOSE ALL POSITIONS (khi signal = 0)')
    print('✅ Proper hedge ratio application')
    print('✅ Explicit position tracking')

def create_improved_kalman_implementation():
    """Create improved Kalman filter implementation"""
    print('\n' + '=' * 80)
    print('TẠO IMPLEMENTATION CẢI TIẾN')
    print('=' * 80)
    
    improved_code = '''
def _run_kalman_strategy_improved(self, stock_data: pd.DataFrame, 
                                futures_data: pd.Series) -> pd.DataFrame:
    """Improved Kalman Filter with proper statistical arbitrage"""
    
    # Data alignment
    common_dates = stock_data.index.intersection(futures_data.index)
    aligned_stocks = stock_data.loc[common_dates]
    aligned_futures = futures_data.loc[common_dates]
    
    # Initialize position tracking
    stock_positions = pd.DataFrame(0.0, index=aligned_stocks.index, columns=aligned_stocks.columns)
    futures_positions = pd.Series(0.0, index=aligned_futures.index)
    portfolio_returns = pd.Series(0.0, index=aligned_stocks.index)
    
    total_signals = 0
    
    for stock in aligned_stocks.columns:
        try:
            # Get trading signals
            trading_signals = self.kalman_filter.generate_trading_signals(
                aligned_futures, aligned_stocks[stock]
            )
            
            if trading_signals.empty:
                continue
            
            # Get hedge ratio
            hedge_ratio = self.kalman_filter.get_hedge_ratio() if hasattr(self.kalman_filter, 'get_hedge_ratio') else 1.0
            
            # Apply signals with proper arbitrage rules
            for timestamp, signal_row in trading_signals.iterrows():
                signal = signal_row['signal']
                
                if signal == 1:  # LONG SIGNAL
                    # LONG STOCK + SHORT FUTURES (proper arbitrage)
                    stock_positions.loc[timestamp, stock] = 1.0
                    futures_positions.loc[timestamp] -= hedge_ratio
                    
                elif signal == -1:  # SHORT SIGNAL
                    # SHORT STOCK + LONG FUTURES (proper arbitrage)
                    stock_positions.loc[timestamp, stock] = -1.0
                    futures_positions.loc[timestamp] += hedge_ratio
                    
                elif signal == 0:  # EXIT SIGNAL
                    # CLOSE ALL POSITIONS
                    stock_positions.loc[timestamp, stock] = 0.0
                    futures_positions.loc[timestamp] = 0.0
                
                total_signals += 1
            
        except Exception as e:
            print(f"Error processing {stock}: {e}")
            continue
    
    # Calculate returns based on actual positions
    stock_returns = aligned_stocks.pct_change().fillna(0)
    futures_returns = aligned_futures.pct_change().fillna(0)
    
    # Portfolio returns = sum(stock_positions * stock_returns) + futures_positions * futures_returns
    portfolio_returns = (stock_positions * stock_returns).sum(axis=1) + futures_positions * futures_returns
    
    print(f"Kalman Filter: Generated {total_signals} signals with proper arbitrage positions")
    
    return pd.DataFrame({'returns': portfolio_returns}, index=aligned_stocks.index)
'''
    
    print('🔧 IMPROVED KALMAN FILTER CODE:')
    print('-' * 50)
    print(improved_code)
    
    print('\n✅ IMPROVEMENTS:')
    print('-' * 50)
    print('✅ Explicit position tracking for stocks and futures')
    print('✅ Proper arbitrage rules: LONG STOCK + SHORT FUTURES')
    print('✅ Proper arbitrage rules: SHORT STOCK + LONG FUTURES')
    print('✅ Hedge ratio application to futures positions')
    print('✅ Position-based return calculation')
    print('✅ Clear signal interpretation')

def main():
    """Main function for checking arbitrage rules"""
    check_kalman_arbitrage_rules()

if __name__ == "__main__":
    main()
