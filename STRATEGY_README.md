# Enhanced Kalman Strategy

## Mô tả
Enhanced Kalman Strategy sử dụng KalmanHedgeFilter để thực hiện statistical arbitrage trên thị trường Việt Nam.

## Cách sử dụng

### 1. Chạy Strategy
```bash
# In-sample period (2021-2022)
python run_strategy.py insample

# Out-of-sample period (2023-2025)
python run_strategy.py outsample
```

### 2. Chạy trực tiếp
```bash
# In-sample period
python backtesting.py insample

# Out-of-sample period
python backtesting.py outsample
```

## Kết quả
- **In-Sample**: 566.03% annual return, 2.23 Sharpe ratio
- **Out-Sample**: 314.98% annual return, 2.35 Sharpe ratio

## Files quan trọng
- `backtesting.py` - Main backtesting engine
- `run_strategy.py` - Simple script to run strategy
- `view_results.py` - View results and open plots
- `filter/kalman_filter.py` - Enhanced KalmanHedgeFilter
- `result/` - Kết quả backtesting và biểu đồ

## Biểu đồ được tạo
- `enhanced_kalman_strategy_analysis_insample.png` - Phân tích in-sample
- `enhanced_kalman_strategy_analysis_outsample.png` - Phân tích out-of-sample

Mỗi biểu đồ bao gồm:
- Cumulative Returns
- Daily Returns
- Rolling Sharpe Ratio
- Drawdown Analysis
- Returns Distribution
- Performance Metrics

## Cấu trúc
```
├── backtesting.py          # Main engine
├── run_strategy.py         # Simple runner
├── filter/
│   └── kalman_filter.py    # Enhanced Kalman filter
├── data/
│   ├── is/                 # In-sample data
│   └── os/                 # Out-of-sample data
├── parameter/              # Configuration files
└── result/                 # Results output
```

## Yêu cầu
- Python 3.8+
- pandas, numpy, matplotlib
- pykalman (for Kalman filter)
