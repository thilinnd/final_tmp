#!/usr/bin/env python3
"""
Test script để kiểm tra optimization parameters
"""

import json
import pandas as pd
import numpy as np
from backtesting_fixed import FixedBacktestingEngine

def test_single_parameter_set():
    """Test với một bộ tham số cụ thể"""
    print("🧪 Testing single parameter set...")
    
    # Tham số test
    test_params = {
        "correlation_threshold": 0.3,
        "max_loss_per_trade": 0.02,
        "take_profit": 0.03,
        "position_size": 0.04,
        "entry_threshold": 0.7,
        "exit_threshold": 0.3
    }
    
    print(f"📊 Test parameters: {test_params}")
    
    try:
        # Tạo custom config
        with open("parameter/in_sample.json", 'r', encoding='utf-8') as f:
            base_config = json.load(f)
        
        # Update với test parameters
        for param_name, param_value in test_params.items():
            if param_name in base_config:
                base_config[param_name] = param_value
        
        print("✅ Custom config created successfully")
        
        # Tạo engine
        engine = FixedBacktestingEngine(custom_config=base_config)
        
        # Chạy backtesting
        print("🚀 Running backtesting...")
        results = engine.run_fixed_arbitrage_strategy("insample")
        
        if results is not None and not results.empty:
            # Tính Sharpe ratio
            returns = results['returns']
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                total_return = (1 + returns).prod() - 1
                max_drawdown = calculate_max_drawdown(returns)
                win_rate = (returns > 0).mean()
                
                print(f"✅ Backtesting completed successfully!")
                print(f"📈 Sharpe Ratio: {sharpe_ratio:.4f}")
                print(f"💰 Total Return: {total_return:.4f}")
                print(f"📉 Max Drawdown: {max_drawdown:.4f}")
                print(f"🎯 Win Rate: {win_rate:.4f}")
                
                return sharpe_ratio
            else:
                print("❌ No valid returns generated")
                return float('-inf')
        else:
            print("❌ No results generated")
            return float('-inf')
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return float('-inf')

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def test_parameter_ranges():
    """Test với các giá trị khác nhau của parameters"""
    print("\n🔍 Testing parameter ranges...")
    
    # Test ranges
    test_cases = [
        {"correlation_threshold": 0.1, "entry_threshold": 0.5, "exit_threshold": 0.2},
        {"correlation_threshold": 0.3, "entry_threshold": 1.0, "exit_threshold": 0.5},
        {"correlation_threshold": 0.5, "entry_threshold": 2.0, "exit_threshold": 1.0},
    ]
    
    results = []
    
    for i, test_params in enumerate(test_cases):
        print(f"\n📊 Test case {i+1}: {test_params}")
        
        try:
            # Load base config
            with open("parameter/in_sample.json", 'r', encoding='utf-8') as f:
                base_config = json.load(f)
            
            # Update with test parameters
            for param_name, param_value in test_params.items():
                if param_name in base_config:
                    base_config[param_name] = param_value
            
            # Create engine
            engine = FixedBacktestingEngine(custom_config=base_config)
            
            # Run backtesting
            results_df = engine.run_fixed_arbitrage_strategy("insample")
            
            if results_df is not None and not results_df.empty:
                returns = results_df['returns']
                if len(returns) > 0 and returns.std() > 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                    total_return = (1 + returns).prod() - 1
                    max_drawdown = calculate_max_drawdown(returns)
                    win_rate = (returns > 0).mean()
                    
                    result = {
                        "test_case": i+1,
                        "parameters": test_params,
                        "sharpe_ratio": sharpe_ratio,
                        "total_return": total_return,
                        "max_drawdown": max_drawdown,
                        "win_rate": win_rate
                    }
                    results.append(result)
                    
                    print(f"✅ Sharpe: {sharpe_ratio:.4f}, Return: {total_return:.4f}")
                else:
                    print("❌ No valid returns")
            else:
                print("❌ No results")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Print summary
    if results:
        print(f"\n📊 SUMMARY OF {len(results)} TEST CASES:")
        print("-" * 60)
        for result in results:
            print(f"Case {result['test_case']}: Sharpe={result['sharpe_ratio']:.4f}, "
                  f"Return={result['total_return']:.4f}, "
                  f"Drawdown={result['max_drawdown']:.4f}")
        
        # Find best case
        best_case = max(results, key=lambda x: x['sharpe_ratio'])
        print(f"\n🏆 BEST CASE: {best_case['test_case']}")
        print(f"   Parameters: {best_case['parameters']}")
        print(f"   Sharpe Ratio: {best_case['sharpe_ratio']:.4f}")
    
    return results

def main():
    """Main function"""
    print("🧪 PARAMETER OPTIMIZATION TEST")
    print("=" * 50)
    
    # Test single parameter set
    sharpe = test_single_parameter_set()
    
    # Test parameter ranges
    results = test_parameter_ranges()
    
    print(f"\n✅ Testing completed!")
    print(f"📁 Results saved in result/ folder")

if __name__ == "__main__":
    main()
