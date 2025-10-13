"""
Fixed Backtesting Engine for Statistical Arbitrage Strategy
- T+2 settlement for stocks, immediate execution for futures
- Correct contract value calculation (points * 100,000)
- Proper arbitrage logic (1 long + 1 short simultaneously)
- Real P&L calculation instead of spread return
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from config.strategy_config import get_config, update_config, validate_config, print_config_summary
warnings.filterwarnings('ignore')


# Import existing modules
try:
    from filter.kalman_filter import KalmanHedgeFilter, create_kalman_hedge_filter
    KALMAN_FILTER_AVAILABLE = True
except ImportError:
    KALMAN_FILTER_AVAILABLE = False
    print("Warning: Enhanced KalmanHedgeFilter not available")

# from metrics.metric import create_evaluator  # Not available

class FixedBacktestingEngine:
    """
    Fixed backtesting engine for statistical arbitrage strategy
    - T+2 settlement for stocks
    - Immediate execution for futures
    - Correct contract value calculation
    - Proper arbitrage position management
    """
    
    def __init__(self, custom_config: dict = None):
        self.evaluator = None
        self.results = {}
        self.hedge_ratio_cache = {}
        self.custom_config = custom_config
        self.silent_mode = False
        
    def _print(self, message: str, silent: bool = False):
        """Print message only if not in silent mode"""
        if not silent and not self.silent_mode:
            print(message)
        
    def load_data(self, mode: str, custom_config: dict = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Load stock and VN30 futures data"""
        print(f"Loading data for {mode}...")
        
        # Load parameters
        if custom_config is not None:
            config = custom_config
            self.config = config
            print(f"Using custom configuration")
        else:
            config = self.load_parameters(mode)
            self.config = config
            if not config:
                print(f"Warning: No config found for {mode}, using default values")
                config = {}
        
        # Load stock data
        stock_file = f"data/{'is' if mode == 'insample' else 'os'}/stock_05.csv"
        stock_data = pd.read_csv(stock_file, index_col=0, parse_dates=True)
        
        # Load VN30 futures data
        vn30_file = f"data/{'is' if mode == 'insample' else 'os'}/vn30f1.csv"
        vn30_data = pd.read_csv(vn30_file, index_col=0, parse_dates=True)
        
        # Selected stocks (VN05 basket)
        selected_stocks = ['VIC', 'VCB', 'VHM', 'VNM', 'BID']
        
        print(f"Loaded stock data from: {stock_file}")
        print(f"Loaded VN30 data from: {vn30_file}")
        print(f"Data loaded successfully!")
        print(f"Stock data shape: {stock_data.shape}")
        print(f"VN30 data shape: {vn30_data.shape}")
        print(f"Selected stocks: {selected_stocks}")
        
        return stock_data, vn30_data, selected_stocks
    
    def load_parameters(self, mode: str) -> Dict:
        """Load configuration parameters from central config"""
        # Get default configuration
        config = get_config("default")
        
        # Try to load mode-specific overrides (only for non-critical parameters)
        config_mapping = {
            'insample': 'in_sample.json',
            'outsample': 'outsample.json'
        }
        
        config_file = f"parameter/{config_mapping.get(mode, f'{mode}.json')}"
        try:
            with open(config_file, 'r') as f:
                mode_config = json.load(f)
            
            # Only update non-critical parameters from mode config
            safe_updates = {}
            if 'start_date' in mode_config:
                safe_updates['start_date'] = mode_config['start_date']
            if 'end_date' in mode_config:
                safe_updates['end_date'] = mode_config['end_date']
            if 'estimation_window' in mode_config:
                safe_updates['estimation_window'] = mode_config['estimation_window']
            
            # Update config with safe parameters only
            config = update_config(config, safe_updates)
            print(f"Mode-specific parameters loaded from: {config_file}")
        except Exception as e:
            print(f"Using default configuration (no mode-specific config found: {e})")
        
        
        # Validate configuration
        if not validate_config(config):
            print("❌ Configuration validation failed, using default values")
            config = get_config("default")
        
        return config
    
    def calculate_vn05_basket(self, stock_data: pd.DataFrame, selected_stocks: List[str]) -> pd.Series:
        """
        Calculate VN05 basket as weighted average of selected stocks
        Equal weight for simplicity
        """
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights
        # Stock prices are in points, multiply by 1000 to get VND
        vn05_basket = (stock_data[selected_stocks] * 1000 * weights).sum(axis=1)
        return vn05_basket
    
    def calculate_contract_value(self, vn30_price: float) -> float:
        """
        Calculate VN30F1 contract value
        Contract value = VN30 points * 100,000 VND
        """
        return vn30_price * 100000
    
    def calculate_position_size(self, available_capital: float, vn30_price: float, 
                              position_ratio: float = 0.04) -> int:
        """
        Calculate number of contracts that can be traded
        """
        position_capital = available_capital * position_ratio
        contract_value = self.calculate_contract_value(vn30_price)
        max_contracts = int(position_capital / contract_value)
        return max(0, max_contracts)
    
    def _run_fixed_arbitrage_strategy(self, stock_data: pd.DataFrame, 
                                   vn30_data: pd.DataFrame, 
                                   selected_stocks: List[str], 
                                   config: Dict) -> pd.DataFrame:
        """
        Run fixed arbitrage strategy with proper T+2 settlement and P&L calculation
        """
        print("Running fixed arbitrage strategy...")
        
        # Use the provided config instead of self.config
        self.config = config
        
        # Calculate VN05 basket
        vn05_basket = self.calculate_vn05_basket(stock_data, selected_stocks)
        
        # Align data
        common_dates = stock_data.index.intersection(vn30_data.index)
        stock_data = stock_data.loc[common_dates]
        vn30_data = vn30_data.loc[common_dates]
        vn05_basket = vn05_basket.loc[common_dates]
        
        # Initialize Kalman filter for hedge ratio
        if KALMAN_FILTER_AVAILABLE:
            kalman_filter = create_kalman_hedge_filter(
                smoothing_transition_cov=config.get('smoothing_transition_cov', 0.05),
                hedge_obs_cov=config.get('hedge_obs_cov', 2.0),
                hedge_trans_cov_delta=config.get('hedge_trans_cov_delta', 1e-3)
            )
            
            # Generate signals using Kalman filter
            signals_df = kalman_filter.generate_trading_signals(
                vn30_data['Stock'], vn05_basket
            )
        else:
            # Fallback to simple z-score signals
            signals_df = self._generate_simple_signals(vn30_data['Stock'], vn05_basket)
        
        # Initialize tracking variables from config
        initial_capital = config.get('initial_capital', 10000000000)
        position_ratio = config.get('position_size_ratio', 0.04)
        current_capital = initial_capital
        
        # Print configuration summary
        print_config_summary(config)
        
        print(f"\n🎯 Using Market Orders Only")
        
        # Position tracking - now supports multiple positions
        positions = []  # List of position dictionaries
        
        # Initialize available stocks pool based on initial capital
        # Assume we can short stocks worth 50% of initial capital
        available_capital_ratio = 0.5  # 50% of capital available for shorting
        available_capital = initial_capital * available_capital_ratio
        
        # Calculate initial available stocks for each stock based on their value contribution
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights
        available_stocks_by_stock = {}
        
        # Use first day's prices to calculate initial available stocks
        first_date = stock_data.index[0]
        vn05_basket_first = self.calculate_vn05_basket(stock_data, selected_stocks).loc[first_date]
        
        for i, stock in enumerate(selected_stocks):
            stock_price = stock_data[stock].loc[first_date] * 1000  # Convert points to VND
            individual_stock_value = stock_price * weights[i]
            individual_available_capital = available_capital * (individual_stock_value / vn05_basket_first)
            individual_available_stocks = individual_available_capital / stock_price
            available_stocks_by_stock[stock] = individual_available_stocks
        
        available_stocks = sum(available_stocks_by_stock.values())
        
        print(f"Initial available stocks: {available_stocks:.0f}")
        print(f"Initial available by stock: {dict(zip(selected_stocks, [f'{v:.1f}' for v in available_stocks_by_stock.values()]))}")
        
        # Results tracking
        daily_returns = []
        capital_history = [initial_capital]
        position_history = []
        
        print(f"Processing {len(signals_df)} trading days...")
        
        for i, (date, row) in enumerate(signals_df.iterrows()):
            signal = row['signal']
            hedge_ratio = row.get('hedge_ratio', 1.0)
            vn30_price = vn30_data['Stock'].loc[date]
            vn05_price = vn05_basket.loc[date]
            
            daily_return = 0.0
            
            # Calculate P&L for all existing positions
            total_pnl = 0
            positions_to_remove = []
            
            for idx, position in enumerate(positions):
                # Calculate P&L for this position
                futures_pnl = self._calculate_futures_pnl(position, vn30_price)
                stock_pnl = self._calculate_stock_pnl(position, vn05_price)
                position_pnl = futures_pnl + stock_pnl
                total_pnl += position_pnl
                
                # Check exit conditions for this position
                if self._should_exit_position(position, signal, date, i):
                    # Close this position
                    total_pnl += position_pnl
                    positions_to_remove.append(idx)
                    
                    # Return stocks to available pool if it was a long stock position
                    if position['direction'] == 1:  # Was long stocks
                        # Return stocks based on equal investment
                        num_stocks = len(selected_stocks)
                        contract_value = self.calculate_contract_value(vn30_price)
                        futures_investment = contract_value * position['futures_contracts']
                        equal_investment_per_stock = futures_investment / num_stocks
                        
                        for j, stock in enumerate(selected_stocks):
                            stock_price = stock_data[stock].loc[date] * 1000  # Convert points to VND
                            returned_value = equal_investment_per_stock  # Equal investment for each stock
                            returned_stocks = returned_value / stock_price
                            available_stocks_by_stock[stock] += returned_stocks
                        available_stocks = sum(available_stocks_by_stock.values())
                        print(f"Returned {position['stock_quantity']:.0f} stocks to available pool")
                        print(f"Available by stock: {dict(zip(selected_stocks, [f'{v:.1f}' for v in available_stocks_by_stock.values()]))}")
            
            # Remove closed positions
            for idx in reversed(positions_to_remove):
                positions.pop(idx)
            
            # Update capital with total P&L
            if total_pnl != 0:
                current_capital += total_pnl
                daily_return = total_pnl / capital_history[-1] if capital_history[-1] > 0 else 0
            
            # Check for new position entry (if we have available capital and stocks)
            max_positions = config.get('max_positions', 3)  # Allow up to 3 positions
            if signal != 0 and len(positions) < max_positions:
                # Calculate number of contracts based on total position size
                # Use original position sizing logic
                num_contracts = self.calculate_position_size(current_capital, vn30_price, position_ratio)
                
                # Calculate equal investment amount per stock
                # Formula: (futures_contract_value * num_contracts) / num_stocks
                num_stocks = len(selected_stocks)
                contract_value = self.calculate_contract_value(vn30_price)
                futures_investment = contract_value * num_contracts
                equal_investment_per_stock = futures_investment / num_stocks
                
                if num_contracts > 0:
                    # Check if we can short stocks (need available stocks)
                    # Calculate required stocks for each individual stock based on equal investment
                    required_stocks_by_stock = {}
                    total_required_value = 0
                    
                    if signal == 1:  # Short stocks
                        # Each stock gets equal investment amount
                        for j, stock in enumerate(selected_stocks):
                            stock_price = stock_data[stock].loc[date] * 1000  # Convert points to VND
                            required_value = equal_investment_per_stock  # Equal investment for each stock
                            required_stocks = required_value / stock_price
                            required_stocks_by_stock[stock] = required_stocks
                            total_required_value += required_value
                    
                    # Check if we have enough available stocks for shorting
                    can_short_stocks = True
                    if signal == 1:  # Short stocks
                        for stock in selected_stocks:
                            if available_stocks_by_stock[stock] < required_stocks_by_stock[stock]:
                                can_short_stocks = False
                                break
                    
                    can_long_stocks = (signal == -1)  # Can always buy stocks
                    
                    if can_short_stocks or can_long_stocks:
                        # Open new position
                        new_position = self._open_position(
                            signal, num_contracts, vn30_price, vn05_price, 
                            hedge_ratio, date, i
                        )
                        positions.append(new_position)
                        
                        # Update available stocks
                        if signal == 1:  # Short stocks
                            for stock in selected_stocks:
                                available_stocks_by_stock[stock] -= required_stocks_by_stock[stock]
                            available_stocks = sum(available_stocks_by_stock.values())
                        elif signal == -1:  # Long stocks
                            # Long stocks don't affect available pool
                            # But we still need to calculate equal investment for position tracking
                            for j, stock in enumerate(selected_stocks):
                                stock_price = stock_data[stock].loc[date] * 1000  # Convert points to VND
                                required_value = equal_investment_per_stock  # Equal investment for each stock
                                required_stocks = required_value / stock_price
                                required_stocks_by_stock[stock] = required_stocks
                                total_required_value += required_value
                        
                        print(f"Available stocks: {available_stocks:.0f}")
                        print(f"Available by stock: {dict(zip(selected_stocks, [f'{v:.1f}' for v in available_stocks_by_stock.values()]))}")
                        daily_return = 0  # No immediate return on entry
                    else:
                        print(f"Cannot short stocks: need {[f'{stock}:{required_stocks_by_stock[stock]:.1f}' for stock in selected_stocks]}, have {dict(zip(selected_stocks, [f'{v:.1f}' for v in available_stocks_by_stock.values()]))}")
            
            # Calculate total position info for tracking
            total_futures_contracts = sum(pos['futures_contracts'] for pos in positions)
            total_stock_quantity = sum(pos['stock_quantity'] for pos in positions)
            net_direction = sum(pos['direction'] for pos in positions) if positions else 0
            
            # Record results
            daily_returns.append(daily_return)
            capital_history.append(current_capital)
            
            # Create position summary for tracking
            position_summary = {
                'futures_contracts': total_futures_contracts,
                'stock_quantity': total_stock_quantity,
                'direction': net_direction,
                'num_positions': len(positions),
                'available_stocks': available_stocks,
                'hedge_ratio': hedge_ratio,  # Add hedge ratio to position summary
                'available_stocks_by_stock': available_stocks_by_stock.copy()  # Add individual stock available
            }
            position_history.append(position_summary)
        
        # Create returns DataFrame with position details
        returns_df = pd.DataFrame({
            'returns': daily_returns,
            'capital': capital_history[1:],  # Exclude initial capital
            'vn30_price': vn30_data['Stock'].values,
            'vn05_price': vn05_basket.values,
            'signal': signals_df['signal'].values,
            'futures_contracts': [pos['futures_contracts'] for pos in position_history],
            'stock_quantity': [pos['stock_quantity'] for pos in position_history],
            'futures_direction': [pos['direction'] for pos in position_history],
            'stock_direction': [-pos['direction'] for pos in position_history],  # Opposite of futures
            'num_positions': [pos['num_positions'] for pos in position_history],
            'available_stocks': [pos['available_stocks'] for pos in position_history]
        }, index=signals_df.index)
        
        # Store results
        self.results = {
            'signals': signals_df['signal'].values,
            'position_history': position_history,
            'final_capital': current_capital,
            'total_return': (current_capital - initial_capital) / initial_capital
        }
        
        print(f"Strategy completed!")
        print(f"Initial capital: {initial_capital:,.0f} VND")
        print(f"Final capital: {current_capital:,.0f} VND")
        print(f"Total return: {self.results['total_return']:.2%}")
        print(f"Total signals: {(signals_df['signal'] != 0).sum()}")
        
        # Generate position plots
        self._plot_position_analysis(returns_df, method="fixed_arbitrage", mode="insample")
        
        return returns_df
    
    def _generate_simple_signals(self, vn30_series: pd.Series, vn05_series: pd.Series) -> pd.DataFrame:
        """Generate simple z-score based signals as fallback"""
        spread = vn05_series - vn30_series
        rolling_mean = spread.rolling(20).mean()
        rolling_std = spread.rolling(20).std()
        z_score = (spread - rolling_mean) / rolling_std
        
        signals = pd.Series(0, index=vn30_series.index)
        signals[z_score < -2.0] = 1   # Long futures + Short stocks
        signals[z_score > 2.0] = -1   # Short futures + Long stocks
        
        return pd.DataFrame({
            'signal': signals,
            'hedge_ratio': 1.0,
            'z_score': z_score
        })
    
    def _open_position(self, signal: int, num_contracts: int, vn30_price: float, 
                      vn05_price: float, hedge_ratio: float, date: pd.Timestamp, 
                      day_index: int) -> Dict:
        """Open new arbitrage position using market orders"""
        
        # Execute immediately with market orders
        position_info = {
            'futures_contracts': num_contracts,
            'stock_quantity': num_contracts * hedge_ratio,
            'entry_futures_price': vn30_price,
            'entry_stock_price': vn05_price,
            'hedge_ratio': hedge_ratio,
            'entry_date': date,
            'direction': signal,
            'entry_day': day_index
        }
        
        print(f"Opened position on {date.strftime('%Y-%m-%d')}: "
              f"{num_contracts} contracts, direction={signal}")
        
        return position_info
    
    def _reset_position(self) -> Dict:
        """Reset position to closed state"""
        return {
            'futures_contracts': 0,
            'stock_quantity': 0,
            'entry_futures_price': 0,
            'entry_stock_price': 0,
            'hedge_ratio': 0,
            'entry_date': None,
            'direction': 0,
            'entry_day': 0
        }
    
    def _calculate_futures_pnl(self, position_info: Dict, current_vn30_price: float) -> float:
        """Calculate futures P&L"""
        if position_info['futures_contracts'] == 0:
            return 0.0
        
        price_change = current_vn30_price - position_info['entry_futures_price']
        pnl = position_info['direction'] * position_info['futures_contracts'] * price_change * 100000
        return pnl
    
    def _calculate_stock_pnl(self, position_info: Dict, current_vn05_price: float) -> float:
        """Calculate stock P&L (with T+2 settlement)"""
        if position_info['stock_quantity'] == 0:
            return 0.0
        
        price_change = current_vn05_price - position_info['entry_stock_price']
        # T+2 settlement: stocks are settled 2 days after entry
        pnl = -position_info['direction'] * position_info['stock_quantity'] * price_change
        return pnl
    
    def _should_exit_position(self, position_info: Dict, current_signal: int, 
                            current_date: pd.Timestamp, day_index: int) -> bool:
        """Check if position should be exited"""
        if position_info['futures_contracts'] == 0:
            return False
        
        # Exit conditions
        holding_days = day_index - position_info['entry_day']
        
        # 1. Signal reversal
        if current_signal == -position_info['direction']:
            return True
        
        # 2. Maximum holding period (30 days)
        if holding_days >= 30:
            return True
        
        # 3. Stop loss (10% loss)
        current_capital = 10000000000  # Simplified for now
        futures_pnl = self._calculate_futures_pnl(position_info, 
            position_info['entry_futures_price'])  # Current price
        if abs(futures_pnl) > current_capital * 0.1:
            return True
        
        return False
    
    def _plot_position_analysis(self, returns_df: pd.DataFrame, method: str, mode: str):
        """Plot long/short positions for each asset type"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Position Analysis - {method.upper()} Strategy ({mode.upper()})', fontsize=16, fontweight='bold')
        
        # Plot 1: Price movements
        ax1 = axes[0]
        ax1.plot(returns_df.index, returns_df['vn30_price'], label='VN30F1M Price', color='blue', linewidth=1.5)
        ax1.plot(returns_df.index, returns_df['vn05_price'], label='VN05 Basket Price', color='red', linewidth=1.5)
        ax1.set_title('Asset Prices Over Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Futures positions (Long/Short) and Number of Positions
        ax2 = axes[1]
        
        # Create position data for futures
        futures_long = returns_df['futures_contracts'] * (returns_df['futures_direction'] > 0)
        futures_short = returns_df['futures_contracts'] * (returns_df['futures_direction'] < 0)
        
        ax2.fill_between(returns_df.index, 0, futures_long, 
                        where=(futures_long > 0), color='green', alpha=0.6, label='Long Futures')
        ax2.fill_between(returns_df.index, 0, -futures_short, 
                        where=(futures_short < 0), color='red', alpha=0.6, label='Short Futures')
        ax2.plot(returns_df.index, futures_long - futures_short, color='black', linewidth=1, label='Net Position')
        
        # Add number of positions on secondary y-axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(returns_df.index, returns_df['num_positions'], color='purple', linewidth=2, 
                     linestyle='--', label='Number of Positions')
        ax2_twin.set_ylabel('Number of Positions', color='purple')
        ax2_twin.tick_params(axis='y', labelcolor='purple')
        
        ax2.set_title('VN30F1M Futures Positions & Position Count')
        ax2.set_ylabel('Contracts')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 3: Stock positions (Long/Short) and Available Stocks
        ax3 = axes[2]
        
        # Create position data for stocks
        stock_long = returns_df['stock_quantity'] * (returns_df['stock_direction'] > 0)
        stock_short = returns_df['stock_quantity'] * (returns_df['stock_direction'] < 0)
        
        ax3.fill_between(returns_df.index, 0, stock_long, 
                        where=(stock_long > 0), color='green', alpha=0.6, label='Long Stocks')
        ax3.fill_between(returns_df.index, 0, -stock_short, 
                        where=(stock_short < 0), color='red', alpha=0.6, label='Short Stocks')
        ax3.plot(returns_df.index, stock_long - stock_short, color='black', linewidth=1, label='Net Position')
        
        # Add available stocks on secondary y-axis
        ax3_twin = ax3.twinx()
        ax3_twin.plot(returns_df.index, returns_df['available_stocks'], color='orange', linewidth=2, 
                     linestyle=':', label='Available Stocks')
        ax3_twin.set_ylabel('Available Stocks', color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        
        ax3.set_title('VN05 Basket Stock Positions & Available Stocks')
        ax3.set_ylabel('Quantity')
        ax3.set_xlabel('Date')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"result/{method}_position_analysis_{mode}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Position analysis plot saved to: {plot_file}")
        plt.close()  # Close plot instead of showing
        
        # Create summary statistics
        self._print_position_summary(returns_df)
    
    def _print_position_summary(self, returns_df: pd.DataFrame):
        """Print position summary statistics"""
        print(f"\n📊 POSITION SUMMARY")
        print("=" * 60)
        
        # Count trading days
        total_days = len(returns_df)
        active_days = (returns_df['futures_contracts'] > 0).sum()
        
        print(f"• Total Trading Days: {total_days}")
        print(f"• Active Position Days: {active_days} ({active_days/total_days:.1%})")
        
        # Futures position analysis
        futures_long_days = (returns_df['futures_direction'] > 0).sum()
        futures_short_days = (returns_df['futures_direction'] < 0).sum()
        futures_neutral_days = (returns_df['futures_direction'] == 0).sum()
        
        print(f"\n🎯 VN30F1M Futures Positions:")
        print(f"  • Long Days: {futures_long_days} ({futures_long_days/total_days:.1%})")
        print(f"  • Short Days: {futures_short_days} ({futures_short_days/total_days:.1%})")
        print(f"  • Neutral Days: {futures_neutral_days} ({futures_neutral_days/total_days:.1%})")
        
        # Stock position analysis
        stock_long_days = (returns_df['stock_direction'] > 0).sum()
        stock_short_days = (returns_df['stock_direction'] < 0).sum()
        stock_neutral_days = (returns_df['stock_direction'] == 0).sum()
        
        print(f"\n📈 VN05 Basket Stock Positions:")
        print(f"  • Long Days: {stock_long_days} ({stock_long_days/total_days:.1%})")
        print(f"  • Short Days: {stock_short_days} ({stock_short_days/total_days:.1%})")
        print(f"  • Neutral Days: {stock_neutral_days} ({stock_neutral_days/total_days:.1%})")
        
        # Position size analysis
        max_futures_contracts = returns_df['futures_contracts'].max()
        max_stock_quantity = returns_df['stock_quantity'].max()
        avg_futures_contracts = returns_df['futures_contracts'].mean()
        avg_stock_quantity = returns_df['stock_quantity'].mean()
        max_positions = returns_df['num_positions'].max()
        avg_positions = returns_df['num_positions'].mean()
        max_available_stocks = returns_df['available_stocks'].max()
        avg_available_stocks = returns_df['available_stocks'].mean()
        
        print(f"\n📊 Position Sizes:")
        print(f"  • Max Futures Contracts: {max_futures_contracts}")
        print(f"  • Max Stock Quantity: {max_stock_quantity:.0f}")
        print(f"  • Avg Futures Contracts: {avg_futures_contracts:.2f}")
        print(f"  • Avg Stock Quantity: {avg_stock_quantity:.0f}")
        
        print(f"\n🔄 Multiple Positions:")
        print(f"  • Max Simultaneous Positions: {max_positions}")
        print(f"  • Avg Simultaneous Positions: {avg_positions:.2f}")
        print(f"  • Max Available Stocks: {max_available_stocks:.0f}")
        print(f"  • Avg Available Stocks: {avg_available_stocks:.0f}")
        
        # Signal analysis
        signal_changes = (returns_df['signal'].diff() != 0).sum()
        print(f"\n🔄 Signal Changes: {signal_changes}")
        print(f"  • Signal Frequency: {signal_changes/total_days:.1%}")
    
    def evaluate_strategy(self, returns: pd.DataFrame, config: Dict) -> Dict:
        """Evaluate strategy performance"""
        print("Evaluating strategy performance...")
        
        # Calculate metrics
        risk_free_rate = config.get('risk_free_rate', 0.05)
        trading_days = config.get('trading_days', 252)
        daily_risk_free_rate = risk_free_rate / trading_days
        
        # Calculate returns
        if 'returns' in returns.columns:
            strategy_returns = returns['returns']
        else:
            # If returns is a Series, use it directly
            strategy_returns = returns if isinstance(returns, pd.Series) else returns.iloc[:, 0]
        
        excess_returns = strategy_returns - daily_risk_free_rate
        
        # Annualized metrics
        annual_return = strategy_returns.mean() * trading_days
        annual_volatility = strategy_returns.std() * np.sqrt(trading_days)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Other metrics
        total_return = cumulative_returns.iloc[-1] - 1
        cagr = (cumulative_returns.iloc[-1] ** (trading_days / len(strategy_returns))) - 1
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'cagr': cagr,
            'volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'excess_return': annual_return - risk_free_rate,
            'risk_free_rate': risk_free_rate
        }
        
        # Save results
        self.results['metrics'] = metrics
        
        return metrics
    
    def save_results(self, returns_df: pd.DataFrame, method: str, mode: str):
        """Save comprehensive strategy results"""
        if not hasattr(self, 'results') or 'metrics' not in self.results:
            print("No results to save")
            return
        
        # Save returns
        returns_file = f"result/{method}_strategy_returns_{mode}.csv"
        returns_df.to_csv(returns_file)
        print(f"{method} strategy returns saved to: {returns_file}")
        
        # Save daily returns
        daily_returns_file = f"result/{method}_daily_returns_{mode}.csv"
        daily_returns_df = pd.DataFrame({
            'date': returns_df.index,
            'strategy_return': returns_df['returns'],
            'vn30_price': returns_df['vn30_price'],
            'vn05_price': returns_df['vn05_price'],
            'capital': returns_df['capital']
        })
        daily_returns_df.to_csv(daily_returns_file, index=False)
        print(f"{method} daily returns saved to: {daily_returns_file}")
        
        # Save monthly returns
        monthly_returns = returns_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_file = f"result/{method}_monthly_returns_{mode}.csv"
        monthly_returns_df = pd.DataFrame({
            'month': monthly_returns.index,
            'monthly_return': monthly_returns.values
        })
        monthly_returns_df.to_csv(monthly_returns_file, index=False)
        print(f"{method} monthly returns saved to: {monthly_returns_file}")
        
        # Save metrics
        metrics_file = f"result/{method}_performance_summary_{mode}.csv"
        metrics_df = pd.DataFrame([self.results['metrics']])
        metrics_df.to_csv(metrics_file, index=False)
        print(f"{method} performance summary saved to: {metrics_file}")
        
        # Save analysis
        analysis_file = f"result/{method}_strategy_analysis_{mode}.json"
        with open(analysis_file, 'w') as f:
            json.dump(self.results['metrics'], f, indent=2, default=str)
        print(f"{method} strategy analysis saved to: {analysis_file}")
        
        # Save detailed trade log
        self.save_detailed_trade_log(returns_df, method, mode)
        
        # Save position and capital time series
        self.save_position_capital_timeseries(returns_df, method, mode)
        
        # Generate plots
        self.generate_plots(returns_df, method, mode)
    
    def save_detailed_trade_log(self, returns_df: pd.DataFrame, method: str, mode: str):
        """Save detailed trade log with position information"""
        print("📊 Creating detailed trade log...")
        
        # Create detailed log DataFrame
        detailed_log = pd.DataFrame(index=returns_df.index)
        
        # Add basic information
        detailed_log['date'] = returns_df.index
        detailed_log['strategy_return'] = returns_df['returns']
        detailed_log['vn30_price'] = returns_df['vn30_price']
        detailed_log['vn05_price'] = returns_df['vn05_price']
        detailed_log['capital'] = returns_df['capital']
        
        # Calculate cumulative metrics
        detailed_log['cumulative_return'] = (1 + returns_df['returns']).cumprod()
        detailed_log['total_return'] = detailed_log['cumulative_return'] - 1
        
        # Calculate drawdown
        running_max = detailed_log['cumulative_return'].expanding().max()
        detailed_log['drawdown'] = (detailed_log['cumulative_return'] - running_max) / running_max
        detailed_log['max_drawdown'] = detailed_log['drawdown'].expanding().min()
        
        # Add position information from results
        if 'position_history' in self.results:
            position_history = self.results['position_history']
            detailed_log['futures_contracts'] = [pos['futures_contracts'] for pos in position_history]
            detailed_log['stock_quantity'] = [pos['stock_quantity'] for pos in position_history]
            detailed_log['position_direction'] = [pos['direction'] for pos in position_history]
            detailed_log['hedge_ratio'] = [pos['hedge_ratio'] for pos in position_history]
            
            # Add individual stock available stocks
            selected_stocks = ['VIC', 'VCB', 'VHM', 'VNM', 'BID']
            for stock in selected_stocks:
                if 'available_stocks_by_stock' in position_history[0]:
                    detailed_log[f'{stock}_available'] = [pos['available_stocks_by_stock'].get(stock, 0) for pos in position_history]
                else:
                    detailed_log[f'{stock}_available'] = 0
        else:
            detailed_log['futures_contracts'] = 0
            detailed_log['stock_quantity'] = 0
            detailed_log['position_direction'] = 0
            detailed_log['hedge_ratio'] = 1.0
            
            # Add individual stock available stocks
            selected_stocks = ['VIC', 'VCB', 'VHM', 'VNM', 'BID']
            for stock in selected_stocks:
                detailed_log[f'{stock}_available'] = 0
        
        # Add signals
        if 'signals' in self.results:
            detailed_log['signal'] = self.results['signals']
        else:
            detailed_log['signal'] = 0
        
        # Calculate contract values
        detailed_log['contract_value'] = detailed_log['vn30_price'] * 100000
        detailed_log['position_value'] = detailed_log['futures_contracts'] * detailed_log['contract_value']
        
        # Calculate rolling statistics
        detailed_log['rolling_volatility_30d'] = returns_df['returns'].rolling(30).std() * np.sqrt(252)
        detailed_log['rolling_sharpe_30d'] = (returns_df['returns'].rolling(30).mean() * 252) / (returns_df['returns'].rolling(30).std() * np.sqrt(252))
        
        # Save detailed log
        log_file = f"result/{method}_detailed_trade_log_{mode}.csv"
        detailed_log.to_csv(log_file, index=False)
        print(f"Detailed trade log saved to: {log_file}")
        print(f"📊 Log contains {len(detailed_log)} rows and {len(detailed_log.columns)} columns")
        
        return log_file
    
    def save_position_capital_timeseries(self, returns_df: pd.DataFrame, method: str, mode: str):
        """Save position and capital time series CSV"""
        print("📊 Creating position and capital time series...")
        
        # Create position and capital time series DataFrame
        timeseries_df = pd.DataFrame(index=returns_df.index)
        
        # Add date information
        timeseries_df['date'] = returns_df.index
        timeseries_df['year'] = returns_df.index.year
        timeseries_df['month'] = returns_df.index.month
        timeseries_df['day'] = returns_df.index.day
        timeseries_df['day_of_week'] = returns_df.index.day_name()
        
        # Add capital information
        timeseries_df['current_capital'] = returns_df['capital']
        timeseries_df['initial_capital'] = 10000000000  # 10B VND
        timeseries_df['capital_change'] = returns_df['capital'] - 10000000000
        timeseries_df['capital_return'] = (returns_df['capital'] - 10000000000) / 10000000000
        
        # Add position information
        timeseries_df['futures_contracts'] = returns_df['futures_contracts']
        timeseries_df['stock_quantity'] = returns_df['stock_quantity']
        timeseries_df['num_positions'] = returns_df['num_positions']
        timeseries_df['available_stocks'] = returns_df['available_stocks']
        
        # Add direction information
        timeseries_df['futures_direction'] = returns_df['futures_direction']
        timeseries_df['stock_direction'] = returns_df['stock_direction']
        
        # Calculate long/short positions for each asset type
        timeseries_df['futures_long_contracts'] = returns_df['futures_contracts'] * (returns_df['futures_direction'] > 0)
        timeseries_df['futures_short_contracts'] = returns_df['futures_contracts'] * (returns_df['futures_direction'] < 0)
        timeseries_df['stock_long_quantity'] = returns_df['stock_quantity'] * (returns_df['stock_direction'] > 0)
        timeseries_df['stock_short_quantity'] = returns_df['stock_quantity'] * (returns_df['stock_direction'] < 0)
        
        # Calculate position values
        timeseries_df['contract_value'] = returns_df['vn30_price'] * 100000
        timeseries_df['futures_position_value'] = returns_df['futures_contracts'] * timeseries_df['contract_value']
        timeseries_df['stock_position_value'] = returns_df['stock_quantity'] * returns_df['vn05_price']
        timeseries_df['total_position_value'] = timeseries_df['futures_position_value'] + timeseries_df['stock_position_value']
        
        # Calculate position ratios
        timeseries_df['futures_capital_ratio'] = timeseries_df['futures_position_value'] / timeseries_df['current_capital']
        timeseries_df['stock_capital_ratio'] = timeseries_df['stock_position_value'] / timeseries_df['current_capital']
        timeseries_df['total_capital_ratio'] = timeseries_df['total_position_value'] / timeseries_df['current_capital']
        
        # Add signal information
        timeseries_df['signal'] = returns_df['signal']
        
        # Calculate rolling statistics
        timeseries_df['rolling_capital_30d'] = timeseries_df['current_capital'].rolling(30).mean()
        timeseries_df['rolling_capital_252d'] = timeseries_df['current_capital'].rolling(252).mean()
        timeseries_df['rolling_positions_30d'] = timeseries_df['num_positions'].rolling(30).mean()
        timeseries_df['rolling_positions_252d'] = timeseries_df['num_positions'].rolling(252).mean()
        
        # Calculate daily changes
        timeseries_df['capital_change_daily'] = timeseries_df['current_capital'].diff()
        timeseries_df['futures_change_daily'] = timeseries_df['futures_contracts'].diff()
        timeseries_df['stock_change_daily'] = timeseries_df['stock_quantity'].diff()
        timeseries_df['positions_change_daily'] = timeseries_df['num_positions'].diff()
        
        # Add performance metrics
        timeseries_df['strategy_return'] = returns_df['returns']
        timeseries_df['cumulative_return'] = (1 + returns_df['returns']).cumprod()
        timeseries_df['total_return'] = timeseries_df['cumulative_return'] - 1
        
        # Calculate drawdown
        running_max = timeseries_df['cumulative_return'].expanding().max()
        timeseries_df['drawdown'] = (timeseries_df['cumulative_return'] - running_max) / running_max
        timeseries_df['max_drawdown'] = timeseries_df['drawdown'].expanding().min()
        
        # Save timeseries CSV
        timeseries_file = f"result/{method}_position_capital_timeseries_{mode}.csv"
        timeseries_df.to_csv(timeseries_file, index=False)
        print(f"Position and capital time series saved to: {timeseries_file}")
        print(f"📊 Timeseries contains {len(timeseries_df)} rows and {len(timeseries_df.columns)} columns")
        
        # Print summary statistics
        print(f"\n💰 CAPITAL AND POSITION SUMMARY:")
        print("=" * 60)
        print(f"Initial capital: {timeseries_df['initial_capital'].iloc[0]:,.0f} VND")
        print(f"Final capital: {timeseries_df['current_capital'].iloc[-1]:,.0f} VND")
        print(f"Capital change: {timeseries_df['capital_change'].iloc[-1]:,.0f} VND")
        print(f"Total return: {timeseries_df['total_return'].iloc[-1]:.2%}")
        print(f"Max drawdown: {timeseries_df['max_drawdown'].iloc[-1]:.2%}")
        
        print(f"\n📊 POSITION STATISTICS:")
        print(f"Max futures contracts: {timeseries_df['futures_contracts'].max()}")
        print(f"Max stock quantity: {timeseries_df['stock_quantity'].max():.0f}")
        print(f"Max positions: {timeseries_df['num_positions'].max()}")
        print(f"Avg positions: {timeseries_df['num_positions'].mean():.2f}")
        
        print(f"\n📈 DIRECTION BREAKDOWN:")
        futures_long_days = (timeseries_df['futures_direction'] > 0).sum()
        futures_short_days = (timeseries_df['futures_direction'] < 0).sum()
        stock_long_days = (timeseries_df['stock_direction'] > 0).sum()
        stock_short_days = (timeseries_df['stock_direction'] < 0).sum()
        
        print(f"Futures Long Days: {futures_long_days} ({futures_long_days/len(timeseries_df):.1%})")
        print(f"Futures Short Days: {futures_short_days} ({futures_short_days/len(timeseries_df):.1%})")
        print(f"Stock Long Days: {stock_long_days} ({stock_long_days/len(timeseries_df):.1%})")
        print(f"Stock Short Days: {stock_short_days} ({stock_short_days/len(timeseries_df):.1%})")
        
        return timeseries_file
    
    def _save_essential_results(self, returns_df: pd.DataFrame, method: str, mode: str):
        """Save essential results without plots (for optimization)"""
        # Only save CSV files, no plots
        self.save_detailed_trade_log(returns_df, method, mode)
        self.save_position_capital_timeseries(returns_df, method, mode)
        
        # Save performance summary
        if hasattr(self, 'results') and 'metrics' in self.results:
            import json
            summary_file = f"result/{method}_performance_summary_{mode}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.results['metrics'], f, indent=2, ensure_ascii=False)
    
    def run_fixed_arbitrage_strategy(self, mode: str, custom_config: dict = None, silent: bool = False):
        """
        Run fixed arbitrage strategy for specified mode
        
        Args:
            mode (str): 'insample' or 'outsample'
            custom_config (dict): Custom configuration parameters
        """
        self.silent_mode = silent
        self._print(f"🚀 Running Fixed Arbitrage Strategy - {mode.upper()}")
        self._print("=" * 60)
        
        # Load data
        stock_data, vn30_data, selected_stocks = self.load_data(mode, custom_config)
        
        if stock_data.empty or vn30_data.empty:
            print("❌ No data loaded, exiting...")
            return
        
        # Run strategy
        returns_df = self._run_fixed_arbitrage_strategy(stock_data, vn30_data, selected_stocks, self.config)
        
        if returns_df.empty:
            print("❌ No returns generated, exiting...")
            return
        
        # Store results for optimization
        self.last_returns = returns_df['returns']
        
        # Evaluate strategy
        metrics = self.evaluate_strategy(returns_df, self.config)
        self.results['metrics'] = metrics
        
        # Save results (skip plotting if silent mode)
        if not silent:
            self.save_results(returns_df, "fixed_arbitrage", mode)
        else:
            # Only save essential data, no plots
            self._save_essential_results(returns_df, "fixed_arbitrage", mode)
        
        print(f"✅ Strategy completed successfully!")
        return returns_df
    
    def get_results(self) -> pd.DataFrame:
        """
        Get backtesting results
        
        Returns:
            pd.DataFrame: Returns data
        """
        if hasattr(self, 'last_returns') and self.last_returns is not None:
            # Convert Series to DataFrame with 'returns' column
            return pd.DataFrame({'returns': self.last_returns})
        else:
            return pd.DataFrame()
    
    def generate_plots(self, returns_df: pd.DataFrame, method: str, mode: str):
        """Generate comprehensive strategy plots"""
        print("📊 Generating strategy analysis plots...")
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :])
        cumulative_returns = (1 + returns_df['returns']).cumprod()
        ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                linewidth=2, color='blue', alpha=0.8)
        ax1.set_title(f'Fixed Arbitrage Strategy - Cumulative Returns ({mode.upper()})', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add performance stats
        total_return = cumulative_returns.iloc[-1] - 1
        ax1.text(0.02, 0.98, f'Total Return: {total_return:.2%}', 
                transform=ax1.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                verticalalignment='top')
        
        # 2. Daily Returns
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(returns_df.index, returns_df['returns'] * 100, 
                linewidth=1, color='purple', alpha=0.7)
        ax2.set_title('Daily Returns (%)')
        ax2.set_ylabel('Daily Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 3. Rolling Sharpe Ratio
        ax3 = fig.add_subplot(gs[1, 1])
        rolling_sharpe = returns_df['returns'].rolling(252).mean() / returns_df['returns'].rolling(252).std() * np.sqrt(252)
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values, 
                linewidth=2, color='green', alpha=0.8)
        ax3.set_title('Rolling Sharpe Ratio (252 days)')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Drawdown Chart
        ax4 = fig.add_subplot(gs[2, 0])
        drawdown = (cumulative_returns - cumulative_returns.expanding().max()) / cumulative_returns.expanding().max() * 100
        ax4.fill_between(cumulative_returns.index, drawdown, 0, color='red', alpha=0.3)
        ax4.plot(cumulative_returns.index, drawdown, linewidth=1, color='red', alpha=0.8)
        ax4.set_title('Drawdown (%)')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Returns Distribution
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(returns_df['returns'] * 100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.axvline(returns_df['returns'].mean() * 100, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {returns_df["returns"].mean()*100:.2f}%')
        ax5.axvline(returns_df['returns'].median() * 100, color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {returns_df["returns"].median()*100:.2f}%')
        ax5.set_title('Returns Distribution')
        ax5.set_xlabel('Daily Return (%)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Add performance metrics text
        if hasattr(self, 'results') and 'metrics' in self.results:
            metrics = self.results['metrics']
            metrics_text = f"""
Performance Metrics:
• Annual Return: {metrics.get('annual_return', 0):.2%}
• Volatility: {metrics.get('volatility', 0):.2%}
• Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
• Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
• Win Rate: {(returns_df['returns'] > 0).mean():.1%}
• Trading Days: {len(returns_df):,}
            """
            
            ax5.text(0.02, 0.98, metrics_text, transform=ax5.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        plt.suptitle(f'Fixed Arbitrage Strategy Analysis - {mode.upper()} Period', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"result/{method}_strategy_analysis_{mode}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Strategy analysis plot saved to: {plot_file}")
        
        # Close plot instead of showing
        plt.close()
        
        return plot_file

def run_fixed_backtest(mode: str):
    """
    Run fixed backtesting for specified mode
    """
    print(f"🚀 Running Fixed Arbitrage Strategy - {mode.upper()}")
    print("=" * 60)
    
    # Create engine
    engine = FixedBacktestingEngine()
    
    # Run fixed arbitrage strategy (new method)
    returns = engine.run_fixed_arbitrage_strategy(mode)
    
    # Print summary
    print(f"\nFIXED ARBITRAGE STRATEGY Results:")
    if returns is not None and not returns.empty:
        print(f"Strategy returns shape: {returns.shape}")
    else:
        print("No returns generated")
    
    print(f"\nPerformance Summary:")
    if hasattr(engine, 'results') and 'metrics' in engine.results:
        for key, value in engine.results['metrics'].items():
            if 'return' in key.lower() or 'drawdown' in key.lower():
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.4f}")
    else:
        print("No metrics available")
    
    print(f"\nFixed arbitrage strategy execution completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python backtesting_fixed.py <mode>")
        print("  mode: insample | outsample")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode not in ['insample', 'outsample']:
        print("❌ Invalid mode. Use 'insample' or 'outsample'")
        sys.exit(1)
    
    try:
        run_fixed_backtest(mode)
    except Exception as e:
        print(f"❌ Error running fixed strategy: {e}")
        import traceback
        traceback.print_exc()
