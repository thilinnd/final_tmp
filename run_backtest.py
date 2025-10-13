#!/usr/bin/env python3
"""
Script đơn giản để chạy backtesting
"""

import sys
from backtesting_fixed import FixedBacktestingEngine

def main():
    """Hàm chính"""
    if len(sys.argv) != 2:
        print("Usage: python run_backtest.py [insample|outsample]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode not in ['insample', 'outsample']:
        print("Error: Mode must be 'insample' or 'outsample'")
        sys.exit(1)
    
    print(f"🚀 Starting backtesting in {mode} mode...")
    
    try:
        # Tạo engine
        engine = FixedBacktestingEngine()
        
        # Chạy backtesting
        engine.run_fixed_arbitrage_strategy(mode)
        
        print(f"✅ Backtesting completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during backtesting: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
