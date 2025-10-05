"""
Comprehensive diagnostic script to find out why OLS strategy has no trades from June 2021
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load and analyze the data for June 2021"""
    print("=" * 80)
    print("DATA LOADING AND ANALYSIS")
    print("=" * 80)
    
    # Load VN30 stocks data
    try:
        vn30_stocks = pd.read_csv('data/vn30_stocks.csv', index_col=0, parse_dates=True)
        print(f"✓ VN30 stocks data loaded: {vn30_stocks.shape}")
        print(f"  Date range: {vn30_stocks.index.min()} to {vn30_stocks.index.max()}")
    except Exception as e:
        print(f"✗ Error loading VN30 stocks: {e}")
        return None, None
    
    # Load VN30F1M data
    try:
        vn30f1m = pd.read_csv('data/vn30f1m.csv', index_col=0, parse_dates=True)
        vn30f1m_price = vn30f1m['price']
        print(f"✓ VN30F1M data loaded: {len(vn30f1m_price)} records")
        print(f"  Date range: {vn30f1m_price.index.min()} to {vn30f1m_price.index.max()}")
    except Exception as e:
        print(f"✗ Error loading VN30F1M: {e}")
        return None, None
    
    # Focus on June 2021
    june_2021_start = pd.to_datetime('2021-06-01')
    june_2021_end = pd.to_datetime('2021-06-30')
    
    june_stocks = vn30_stocks[(vn30_stocks.index >= june_2021_start) & (vn30_stocks.index <= june_2021_end)]
    june_futures = vn30f1m_price[(vn30f1m_price.index >= june_2021_start) & (vn30f1m_price.index <= june_2021_end)]
    
    print(f"\nJune 2021 Data Analysis:")
    print(f"  VN30 stocks records: {len(june_stocks)}")
    print(f"  VN30F1M records: {len(june_futures)}")
    print(f"  Common dates: {len(june_stocks.index.intersection(june_futures.index))}")
    
    return june_stocks, june_futures

def analyze_ols_requirements():
    """Analyze OLS estimation requirements"""
    print("\n" + "=" * 80)
    print("OLS ESTIMATION REQUIREMENTS ANALYSIS")
    print("=" * 80)
    
    june_stocks, june_futures = load_and_analyze_data()
    
    if june_stocks is None or june_futures is None:
        return
    
    # Check minimum periods requirement
    min_periods = 30  # From OLS estimator
    print(f"Minimum periods required for OLS: {min_periods}")
    
    # Check data availability
    print(f"Available data in June 2021:")
    print(f"  VN30 stocks: {len(june_stocks)} days")
    print(f"  VN30F1M: {len(june_futures)} days")
    
    # Check if we have enough data for estimation
    if len(june_stocks) < min_periods:
        print(f"✗ Insufficient VN30 stocks data for OLS estimation ({len(june_stocks)} < {min_periods})")
    else:
        print(f"✓ Sufficient VN30 stocks data for OLS estimation")
    
    if len(june_futures) < min_periods:
        print(f"✗ Insufficient VN30F1M data for OLS estimation ({len(june_futures)} < {min_periods})")
    else:
        print(f"✓ Sufficient VN30F1M data for OLS estimation")
    
    # Check for data alignment
    common_dates = june_stocks.index.intersection(june_futures.index)
    print(f"\nCommon trading dates in June 2021: {len(common_dates)}")
    
    if len(common_dates) < min_periods:
        print(f"✗ Insufficient aligned data for OLS estimation ({len(common_dates)} < {min_periods})")
    else:
        print(f"✓ Sufficient aligned data for OLS estimation")

def test_ols_estimation():
    """Test OLS estimation with June 2021 data"""
    print("\n" + "=" * 80)
    print("OLS ESTIMATION TESTING")
    print("=" * 80)
    
    june_stocks, june_futures = load_and_analyze_data()
    
    if june_stocks is None or june_futures is None:
        return
    
    # Import OLS estimator
    try:
        from filter.ols_estimation import OLSEstimator, create_ols_estimator
        print("✓ OLS estimator imported successfully")
    except Exception as e:
        print(f"✗ Error importing OLS estimator: {e}")
        return
    
    # Create OLS estimator
    ols_estimator = create_ols_estimator(
        min_periods=30,
        confidence_level=0.05,
        rolling_window=20,
        max_window=252
    )
    
    # Align data
    common_dates = june_stocks.index.intersection(june_futures.index)
    if len(common_dates) < 30:
        print(f"✗ Insufficient common dates for OLS testing ({len(common_dates)} < 30)")
        return
    
    aligned_stocks = june_stocks.loc[common_dates]
    aligned_futures = june_futures.loc[common_dates]
    
    print(f"\nTesting OLS estimation with {len(common_dates)} aligned dates:")
    
    # Test OLS estimation for each stock
    for i, stock in enumerate(aligned_stocks.columns[:5]):  # Test first 5 stocks
        try:
            print(f"\nTesting {stock}:")
            
            # Calculate OLS hedge ratio
            ols_result = ols_estimator.calculate_hedge_ratio(
                aligned_futures, aligned_stocks[stock], method='ols'
            )
            
            print(f"  Hedge ratio: {ols_result['hedge_ratio']:.4f}")
            print(f"  R-squared: {ols_result['r_squared']:.4f}")
            print(f"  P-value: {ols_result['p_value']:.4f}")
            print(f"  Cointegration p-value: {ols_result['cointegration_pvalue']:.4f}")
            print(f"  ADF p-value: {ols_result['adf_pvalue']:.4f}")
            print(f"  Half-life: {ols_result['half_life']}")
            print(f"  Is cointegrated: {ols_result['is_cointegrated']}")
            print(f"  Error: {ols_result.get('error', 'None')}")
            
            if ols_result['is_cointegrated']:
                print(f"  ✓ {stock} is cointegrated with VN30F1M")
            else:
                print(f"  ✗ {stock} is NOT cointegrated with VN30F1M")
                
        except Exception as e:
            print(f"  ✗ Error testing {stock}: {e}")

def test_trading_signals():
    """Test trading signal generation"""
    print("\n" + "=" * 80)
    print("TRADING SIGNAL GENERATION TESTING")
    print("=" * 80)
    
    june_stocks, june_futures = load_and_analyze_data()
    
    if june_stocks is None or june_futures is None:
        return
    
    # Import OLS estimator
    try:
        from filter.ols_estimation import OLSEstimator, create_ols_estimator
        ols_estimator = create_ols_estimator()
    except Exception as e:
        print(f"✗ Error importing OLS estimator: {e}")
        return
    
    # Align data
    common_dates = june_stocks.index.intersection(june_futures.index)
    if len(common_dates) < 30:
        print(f"✗ Insufficient common dates for signal testing ({len(common_dates)} < 30)")
        return
    
    aligned_stocks = june_stocks.loc[common_dates]
    aligned_futures = june_futures.loc[common_dates]
    
    print(f"\nTesting trading signal generation with {len(common_dates)} aligned dates:")
    
    # Test signal generation for each stock
    for i, stock in enumerate(aligned_stocks.columns[:5]):  # Test first 5 stocks
        try:
            print(f"\nTesting signals for {stock}:")
            
            # Calculate spread with dynamic window
            spread_data = ols_estimator.calculate_spread_with_dynamic_window(
                aligned_futures, aligned_stocks[stock]
            )
            
            if spread_data.empty:
                print(f"  ✗ No spread data generated for {stock}")
                continue
            
            print(f"  Spread data shape: {spread_data.shape}")
            print(f"  Z-score range: [{spread_data['z_score'].min():.3f}, {spread_data['z_score'].max():.3f}]")
            print(f"  Z-score mean: {spread_data['z_score'].mean():.3f}")
            print(f"  Z-score std: {spread_data['z_score'].std():.3f}")
            
            # Generate trading signals
            signals = ols_estimator.generate_trading_signals(
                spread_data, 
                entry_threshold=2.0, 
                exit_threshold=0.5
            )
            
            if signals.empty:
                print(f"  ✗ No signals generated for {stock}")
                continue
            
            # Count signals
            long_signals = (signals['signal'] == 1).sum()
            short_signals = (signals['signal'] == -1).sum()
            exit_signals = (signals['signal'] == 0).sum()
            total_signals = long_signals + short_signals + exit_signals
            
            print(f"  Long signals: {long_signals}")
            print(f"  Short signals: {short_signals}")
            print(f"  Exit signals: {exit_signals}")
            print(f"  Total signals: {total_signals}")
            
            if total_signals > 0:
                print(f"  ✓ Signals generated for {stock}")
            else:
                print(f"  ✗ No signals generated for {stock}")
                
        except Exception as e:
            print(f"  ✗ Error testing signals for {stock}: {e}")

def analyze_strategy_logic():
    """Analyze the strategy logic in backtesting.py"""
    print("\n" + "=" * 80)
    print("STRATEGY LOGIC ANALYSIS")
    print("=" * 80)
    
    # Read the backtesting.py file to understand the OLS strategy logic
    try:
        with open('backtesting.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for OLS strategy implementation
        if '_run_ols_strategy' in content:
            print("✓ Found _run_ols_strategy method in backtesting.py")
            
            # Extract the OLS strategy method
            start_idx = content.find('def _run_ols_strategy')
            if start_idx != -1:
                # Find the end of the method
                lines = content[start_idx:].split('\n')
                method_lines = []
                indent_level = None
                
                for line in lines:
                    if line.strip().startswith('def _run_ols_strategy'):
                        method_lines.append(line)
                        # Get the indentation level
                        indent_level = len(line) - len(line.lstrip())
                        continue
                    
                    if line.strip() and not line.startswith(' ' * (indent_level + 1)) and not line.startswith('\t'):
                        break
                    
                    method_lines.append(line)
                
                print("\nOLS Strategy Method:")
                print("-" * 50)
                for line in method_lines[:20]:  # Show first 20 lines
                    print(line)
                print("...")
        else:
            print("✗ _run_ols_strategy method not found in backtesting.py")
            
    except Exception as e:
        print(f"✗ Error reading backtesting.py: {e}")

def check_data_quality():
    """Check data quality issues"""
    print("\n" + "=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)
    
    june_stocks, june_futures = load_and_analyze_data()
    
    if june_stocks is None or june_futures is None:
        return
    
    # Check for missing values
    print(f"Missing values in June 2021:")
    print(f"  VN30 stocks: {june_stocks.isnull().sum().sum()}")
    print(f"  VN30F1M: {june_futures.isnull().sum()}")
    
    # Check for extreme values
    print(f"\nVN30F1M price analysis:")
    print(f"  Min: {june_futures.min():.2f}")
    print(f"  Max: {june_futures.max():.2f}")
    print(f"  Mean: {june_futures.mean():.2f}")
    print(f"  Std: {june_futures.std():.2f}")
    
    # Check for extreme price movements
    june_returns = june_futures.pct_change().dropna()
    extreme_returns = june_returns[abs(june_returns) > 0.1]  # >10% daily moves
    print(f"  Extreme daily moves (>10%): {len(extreme_returns)}")
    
    if len(extreme_returns) > 0:
        print("  Extreme move dates:")
        for date, return_val in extreme_returns.items():
            print(f"    {date}: {return_val:.2%}")
    
    # Check correlation
    print(f"\nCorrelation analysis:")
    for i, stock in enumerate(june_stocks.columns[:5]):
        try:
            corr = june_stocks[stock].corr(june_futures)
            print(f"  {stock} vs VN30F1M: {corr:.4f}")
        except Exception as e:
            print(f"  {stock} vs VN30F1M: Error - {e}")

def main():
    """Main diagnostic function"""
    print("OLS STRATEGY DIAGNOSTIC - NO TRADES FROM JUNE 2021")
    print("=" * 80)
    
    # Run all diagnostic tests
    load_and_analyze_data()
    analyze_ols_requirements()
    test_ols_estimation()
    test_trading_signals()
    analyze_strategy_logic()
    check_data_quality()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    print("""
    POTENTIAL REASONS FOR NO TRADES FROM JUNE 2021:
    
    1. DATA ISSUES:
       - Insufficient data points for OLS estimation
       - Missing or misaligned data between stocks and futures
       - Extreme price movements causing data quality issues
    
    2. CORRELATION ISSUES:
       - Stocks may not meet correlation threshold (0.6) with VN30F1M
       - Insufficient cointegration between pairs
    
    3. SIGNAL GENERATION ISSUES:
       - Z-scores may not exceed entry/exit thresholds
       - Spread may not show sufficient mean reversion
       - Volatility too low to generate meaningful signals
    
    4. STRATEGY LOGIC ISSUES:
       - OLS estimation may be failing
       - Trading signal generation may have bugs
       - Position management constraints
    
    5. CONFIGURATION ISSUES:
       - Entry/exit thresholds too strict
       - Minimum periods requirement too high
       - Correlation threshold too restrictive
    
    RECOMMENDATIONS:
    
    1. Check data quality and alignment
    2. Reduce minimum periods requirement
    3. Lower correlation threshold
    4. Adjust entry/exit thresholds
    5. Add debug logging to OLS strategy
    6. Implement fallback strategies
    """)

if __name__ == "__main__":
    main()
