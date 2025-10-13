#!/usr/bin/env python3
"""
Central Strategy Configuration
All strategy parameters are defined here in one place
"""

# =============================================================================
# CORE STRATEGY PARAMETERS
# =============================================================================

# Capital Management
INITIAL_CAPITAL = 10000000000  # 10 tỷ VND
POSITION_SIZE_RATIO = 0.04     # 4% vốn mỗi vị thế
MAX_POSITIONS = 1              # 1 vị thế arbitrage (1 long + 1 short)
TRANSACTION_COST = 0.001       # 0.1% phí giao dịch
RISK_FREE_RATE = 0.05          # 5% lãi suất phi rủi ro

# Trading Thresholds
ENTRY_THRESHOLD = 2.0          # Ngưỡng vào lệnh (số độ lệch chuẩn)
EXIT_THRESHOLD = 0.5           # Ngưỡng thoát lệnh
STOP_LOSS = 0.05               # Stop loss 5%
TAKE_PROFIT = 0.03             # Take profit 3%

# Kalman Filter Configuration
KALMAN_CONFIG = {
    "initial_state_mean": [1.0, 0.0],
    "initial_state_covariance": [[1.0, 0.0], [0.0, 1.0]],
    "observation_covariance": 0.01,
    "transition_covariance": [[0.01, 0.0], [0.0, 0.01]],
    "window_size": 30,
    "min_periods": 10
}

# Data Configuration
SELECTED_STOCKS = ['VIC', 'VCB', 'VHM', 'VNM', 'BID']  # VN05 basket

# =============================================================================
# DERIVED PARAMETERS (calculated from core parameters)
# =============================================================================

def get_position_capital(initial_capital: float = None) -> float:
    """Calculate position capital based on initial capital and position size ratio"""
    if initial_capital is None:
        initial_capital = INITIAL_CAPITAL
    return initial_capital * POSITION_SIZE_RATIO

def get_max_contracts(vn30_price: float, initial_capital: float = None) -> int:
    """Calculate maximum number of contracts based on position capital"""
    position_capital = get_position_capital(initial_capital)
    contract_value = vn30_price * 100000  # VN30 points * 100,000 VND
    return max(0, int(position_capital / contract_value))

# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

CONSERVATIVE_CONFIG = {
    "initial_capital": INITIAL_CAPITAL,
    "position_size_ratio": 0.02,  # 2%
    "max_positions": 1,
    "transaction_cost": TRANSACTION_COST,
    "risk_free_rate": RISK_FREE_RATE,
    "trading_thresholds": {
        "entry_threshold": 2.5,
        "exit_threshold": 0.3,
        "stop_loss": 0.03,
        "take_profit": 0.02
    },
    "kalman_config": KALMAN_CONFIG
}

AGGRESSIVE_CONFIG = {
    "initial_capital": INITIAL_CAPITAL,
    "position_size_ratio": 0.06,  # 6%
    "max_positions": 1,
    "transaction_cost": TRANSACTION_COST,
    "risk_free_rate": RISK_FREE_RATE,
    "trading_thresholds": {
        "entry_threshold": 1.5,
        "exit_threshold": 0.8,
        "stop_loss": 0.08,
        "take_profit": 0.05
    },
    "kalman_config": KALMAN_CONFIG
}

HIGH_FREQUENCY_CONFIG = {
    "initial_capital": INITIAL_CAPITAL,
    "position_size_ratio": POSITION_SIZE_RATIO,
    "max_positions": 1,
    "transaction_cost": TRANSACTION_COST,
    "risk_free_rate": RISK_FREE_RATE,
    "trading_thresholds": {
        "entry_threshold": 1.0,
        "exit_threshold": 0.2,
        "stop_loss": 0.02,
        "take_profit": 0.015
    },
    "kalman_config": {
        **KALMAN_CONFIG,
        "window_size": 15,
        "min_periods": 5
    }
}

# Default configuration (optimized based on analysis)
DEFAULT_CONFIG = {
    "initial_capital": INITIAL_CAPITAL,
    "position_size_ratio": 0.015,  # Optimized: 1.5% (was 4%)
    "max_positions": 3,  # Allow up to 3 positions simultaneously
    "transaction_cost": TRANSACTION_COST,
    "risk_free_rate": RISK_FREE_RATE,
    "trading_thresholds": {
        "entry_threshold": 2.8,    # Optimized: 2.8σ (was 2.0σ)
        "exit_threshold": 0.2,     # Optimized: 0.2σ (was 0.5σ)
        "stop_loss": 0.02,         # Optimized: 2% (was 5%)
        "take_profit": 0.012       # Optimized: 1.2% (was 3%)
    },
    "kalman_config": {
        "initial_state_mean": [1.0, 0.0],
        "initial_state_covariance": [[1.0, 0.0], [0.0, 1.0]],
        "observation_covariance": 0.003,  # Optimized: 0.003 (was 0.01)
        "transition_covariance": [[0.003, 0.0], [0.0, 0.003]],  # Optimized
        "window_size": 25,         # Optimized: 25 days (was 30)
        "min_periods": 12          # Optimized: 12 days (was 10)
    }
}

# =============================================================================
# CONFIGURATION MANAGEMENT FUNCTIONS
# =============================================================================

def get_config(preset: str = "default") -> dict:
    """Get configuration by preset name"""
    configs = {
        "default": DEFAULT_CONFIG,
        "conservative": CONSERVATIVE_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
        "high_frequency": HIGH_FREQUENCY_CONFIG
    }
    return configs.get(preset, DEFAULT_CONFIG).copy()

def update_config(base_config: dict, updates: dict) -> dict:
    """Update configuration with new values"""
    config = base_config.copy()
    
    # Deep update for nested dictionaries
    for key, value in updates.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    
    return config

def validate_config(config: dict) -> bool:
    """Validate configuration parameters"""
    required_keys = [
        'initial_capital', 'position_size_ratio', 'max_positions',
        'transaction_cost', 'risk_free_rate', 'trading_thresholds', 'kalman_config'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required config key: {key}")
            return False
    
    # Validate trading thresholds
    thresholds = config.get('trading_thresholds', {})
    required_thresholds = ['entry_threshold', 'exit_threshold', 'stop_loss', 'take_profit']
    for key in required_thresholds:
        if key not in thresholds:
            print(f"❌ Missing required threshold: {key}")
            return False
    
    return True

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def print_config_summary(config: dict = None):
    """Print configuration summary"""
    if config is None:
        config = DEFAULT_CONFIG
    
    print("📊 STRATEGY CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"💰 Initial Capital: {config['initial_capital']:,.0f} VND")
    print(f"📊 Position Size: {config['position_size_ratio']:.1%}")
    print(f"📦 Max Positions: {config['max_positions']}")
    print(f"💸 Transaction Cost: {config['transaction_cost']:.1%}")
    print(f"📈 Risk Free Rate: {config['risk_free_rate']:.1%}")
    
    thresholds = config['trading_thresholds']
    print(f"\n🎯 Trading Thresholds:")
    print(f"  • Entry: {thresholds['entry_threshold']:.1f}σ")
    print(f"  • Exit: {thresholds['exit_threshold']:.1f}σ")
    print(f"  • Stop Loss: {thresholds['stop_loss']:.1%}")
    print(f"  • Take Profit: {thresholds['take_profit']:.1%}")
    
    kalman = config['kalman_config']
    print(f"\n🔧 Kalman Filter:")
    print(f"  • Window Size: {kalman['window_size']} days")
    print(f"  • Min Periods: {kalman['min_periods']} days")

if __name__ == "__main__":
    print_config_summary()
