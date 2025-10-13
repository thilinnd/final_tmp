#!/usr/bin/env python3
"""
Script để chạy parameter optimization
"""

import sys
import os
from optimization import StrategyOptimizer

def main():
    """Hàm chính"""
    print("🔍 PARAMETER OPTIMIZATION FOR STATISTICAL ARBITRAGE STRATEGY")
    print("=" * 70)
    
    try:
        # Kiểm tra file config
        if not os.path.exists("parameter/optimization_parameter.json"):
            print("❌ Error: optimization_parameter.json not found")
            sys.exit(1)
        
        if not os.path.exists("parameter/in_sample.json"):
            print("❌ Error: in_sample.json not found")
            sys.exit(1)
        
        # Tạo optimizer
        print("📊 Loading optimization configuration...")
        optimizer = StrategyOptimizer()
        
        # Chạy optimization
        print("🚀 Starting optimization...")
        study = optimizer.optimize()
        
        print("\n✅ Optimization completed successfully!")
        print("📁 Results saved in result/optimization/")
        
    except KeyboardInterrupt:
        print("\n⏹️ Optimization interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
