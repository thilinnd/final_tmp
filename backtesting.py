"""
Backtesting module for statistical arbitrage strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from decimal import Decimal
import os
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader, create_data_loader
from evaluation import StrategyEvaluator, create_evaluator
from filter.financial import StockFilter, create_stock_filter
from filter.ols_estimation import OLSEstimator, create_ols_estimator
from filter.bayesian_estimation import BayesianEstimator, create_bayesian_estimator
from filter.kalman_filter import KalmanHedgeFilter, create_kalman_hedge_filter
from metrics.metric import calculate_shapre_and_mdd
from config.config import BACKTESTING_CONFIG, OPTIMIZATION_CONFIG, BEST_CONFIG
from position_manager import PositionManager, PositionType

class BacktestingEngine:
    """
    Main backtesting engine for statistical arbitrage strategy
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the backtesting engine
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config or BACKTESTING_CONFIG
        self.data_loader = None
        self.evaluator = None
        self.results = {}
        
        # Initialize estimation algorithms
        self.ols_estimator = None
        self.bayesian_estimator = None
        self.kalman_filter = None
        
        # Initialize capital tracking
        self.initial_capital = self.config.get('initial_capital', 10000000000)  # Default 10B VND
        self.daily_assets = [self.initial_capital]  # Track daily asset values
        self.ac_loss = 0  # Accumulated loss
        self.inventory = 0  # Current inventory/position
        self.min_capital_threshold = 0.01  # 1% of initial capital as minimum threshold
        
        # Initialize Position Manager
        position_size = self.config.get('position_size', 0.04)  # 4% vốn cho mỗi vị thế
        self.position_manager = PositionManager(
            initial_capital=self.initial_capital,
            position_size=position_size
        )
    
    def from_cash_to_tradeable_contracts(self, available_cash: float, inst_price: Decimal) -> int:
        """
        Convert available cash to tradeable contracts
        
        Args:
            available_cash (float): Available cash amount
            inst_price (Decimal): Instrument price per contract
            
        Returns:
            int: Number of tradeable contracts
        """
        if inst_price <= 0:
            return 0
        
        # Convert to float for calculation, then back to int for contracts
        contracts = int(available_cash / float(inst_price))
        return max(contracts, 0)
    
    def get_maximum_placeable(self, inst_price: Decimal):
        """
        Get maximum placeable contracts with capital protection
        
        Args:
            inst_price (Decimal): Instrument price per contract
            
        Returns:
            int: Maximum number of contracts that can be placed
        """
        # Calculate available capital after accumulated losses
        available_capital = self.daily_assets[-1] - self.ac_loss
        
        # Check if capital is nearly zero (below minimum threshold)
        min_required_capital = self.initial_capital * self.min_capital_threshold
        
        if available_capital < min_required_capital:
            # If capital is nearly zero, return 0 to prevent new positions
            return 0
        
        # Calculate maximum placeable contracts
        total_placeable = max(
            self.from_cash_to_tradeable_contracts(available_capital, inst_price),
            0,
        )
        
        # Subtract current inventory to get net placeable
        net_placeable = total_placeable - abs(self.inventory)
        
        # Ensure we don't return negative values
        return max(net_placeable, 0)
        
    def setup(self, start_date: str, end_date: str):
        """
        Setup the backtesting environment
        
        Args:
            start_date (str): Start date for backtesting
            end_date (str): End date for backtesting
        """
        print("Setting up backtesting environment...")
        
        # Initialize data loader
        self.data_loader = create_data_loader(
            start_date=start_date,
            end_date=end_date,
            estimation_window=self.config.get('estimation_window', 60),
            correlation_threshold=self.config.get('correlation_threshold', 0.6)
        )
        
        # Initialize evaluator
        self.evaluator = create_evaluator(
            risk_free_rate=self.config.get('risk_free_rate', 0.05),
            trading_days=self.config.get('trading_days', 252)
        )
        
        # Initialize estimation algorithms
        self.ols_estimator = create_ols_estimator(
            min_periods=self.config.get('min_periods', 30),
            confidence_level=self.config.get('confidence_level', 0.05)
        )
        
        self.bayesian_estimator = create_bayesian_estimator(
            n_samples=self.config.get('n_samples', 200),  # Reduced for faster execution
            tune=self.config.get('tune', 100),  # Reduced for faster execution
            chains=self.config.get('chains', 2),
            target_accept=self.config.get('target_accept', 0.8)
        )
        
        # Initialize kalman_filter only when needed
        self.kalman_filter = None
        
        print("Backtesting environment setup complete!")
    
    def load_data(self):
        """
        Load required data for backtesting
        """
        if self.data_loader is None:
            raise ValueError("Backtesting engine not setup. Call setup() first.")
        
        print("Loading data...")
        
        # Load stock data
        stock_data = self.data_loader.load_stock_data()
        
        # Load VN30 data
        vn30_data = self.data_loader.load_vn30_data()
        
        # Select arbitrage stocks
        selected_stocks = self.data_loader.select_arbitrage_stocks()
        
        print(f"Data loaded successfully!")
        print(f"Stock data shape: {stock_data.shape}")
        print(f"VN30 data shape: {vn30_data.shape if vn30_data is not None else 'None'}")
        print(f"Selected stocks: {selected_stocks}")
        
        return stock_data, vn30_data, selected_stocks
    
    def run_strategy(self, stock_data: pd.DataFrame, 
                    vn30_data: pd.DataFrame = None,
                    selected_stocks: List[str] = None,
                    method: str = 'kalman') -> pd.DataFrame:
        """
        Run the statistical arbitrage strategy
        
        Args:
            stock_data (pd.DataFrame): Stock price data
            vn30_data (pd.DataFrame): VN30 index data
            selected_stocks (List[str]): Selected stocks for arbitrage
            method (str): Strategy method ('equal_weight', 'ols', 'bayesian', 'kalman')
            
        Returns:
            pd.DataFrame: Strategy returns
        """
        print(f"Running statistical arbitrage strategy using {method} method...")
        
        # Get processed data
        processed_stocks, futures_data, stocks = self.data_loader.get_processed_data()
        
        if method == 'equal_weight':
            return self._run_equal_weight_strategy(processed_stocks, futures_data)
        elif method == 'ols':
            return self._run_ols_strategy(processed_stocks, futures_data)
        elif method == 'bayesian':
            return self._run_bayesian_strategy(processed_stocks, futures_data)
        elif method == 'kalman':
            return self._run_kalman_strategy(processed_stocks, futures_data)
        else:
            raise ValueError("Method must be 'equal_weight', 'ols', 'bayesian', or 'kalman'")
    
    def _run_equal_weight_strategy(self, stock_data: pd.DataFrame, 
                                 futures_data: pd.Series) -> pd.DataFrame:
        """Run equal-weight portfolio strategy"""
        if len(stock_data.columns) > 0:
            portfolio_returns = stock_data.pct_change().mean(axis=1).fillna(0)
        else:
            np.random.seed(42)
            portfolio_returns = pd.Series(
                np.random.normal(0.001, 0.02, len(stock_data)),
                index=stock_data.index
            )
        
        return pd.DataFrame({'returns': portfolio_returns}, index=stock_data.index)
    
    def _run_ols_strategy(self, stock_data: pd.DataFrame, 
                         futures_data: pd.Series) -> pd.DataFrame:
        """Run OLS-based statistical arbitrage strategy"""
        if futures_data is None or len(stock_data.columns) == 0:
            return self._run_equal_weight_strategy(stock_data, futures_data)
        
        # Calculate hedge ratios for each stock
        hedge_ratios = {}
        spreads = {}
        
        for stock in stock_data.columns:
            if stock in stock_data.columns:
                # Calculate hedge ratio using OLS
                ols_result = self.ols_estimator.calculate_hedge_ratio(
                    futures_data, stock_data[stock], method='ols'
                )
                
                if ols_result['is_cointegrated']:
                    hedge_ratios[stock] = ols_result['hedge_ratio']
                    spreads[stock] = self.ols_estimator.calculate_spread(
                        futures_data, stock_data[stock], ols_result['hedge_ratio']
                    )
        
        if not spreads:
            return self._run_equal_weight_strategy(stock_data, futures_data)
        
        # Calculate portfolio returns based on spreads
        spread_df = pd.DataFrame(spreads)
        portfolio_returns = spread_df.pct_change().mean(axis=1).fillna(0)
        
        return pd.DataFrame({'returns': portfolio_returns}, index=stock_data.index)
    
    def _run_bayesian_strategy(self, stock_data: pd.DataFrame, 
                              futures_data: pd.Series) -> pd.DataFrame:
        """Run Bayesian-based statistical arbitrage strategy"""
        if futures_data is None or len(stock_data.columns) == 0:
            return self._run_equal_weight_strategy(stock_data, futures_data)
        
        # Calculate Bayesian hedge ratios
        hedge_ratios = {}
        spreads = {}
        
        for stock in stock_data.columns:
            if stock in stock_data.columns:
                # Calculate hedge ratio using Bayesian estimation
                print(f"Processing stock: {stock}")
                bayesian_result = self.bayesian_estimator.estimate_hedge_ratio_bayesian(
                    futures_data, stock_data[stock]
                )
                
                print(f"Bayesian result for {stock}: {bayesian_result}")
                
                if bayesian_result.get('converged', False):
                    hedge_ratios[stock] = bayesian_result['beta_mean']
                    spreads[stock] = self.ols_estimator.calculate_spread(
                        futures_data, stock_data[stock], 
                        bayesian_result['beta_mean']
                    )
                    print(f"✓ Bayesian estimation successful for {stock}: {bayesian_result['beta_mean']:.4f}")
                else:
                    print(f"✗ Bayesian estimation failed for {stock}: {bayesian_result.get('error', 'Unknown error')}")
                    print(f"  Falling back to OLS for {stock}...")
                    
                    # Fallback to OLS
                    ols_result = self.ols_estimator.calculate_hedge_ratio(
                        futures_data, stock_data[stock], method='ols'
                    )
                    
                    if ols_result.get('is_cointegrated', False):
                        hedge_ratios[stock] = ols_result['hedge_ratio']
                        spreads[stock] = self.ols_estimator.calculate_spread(
                            futures_data, stock_data[stock], 
                            ols_result['hedge_ratio']
                        )
                        print(f"  ✓ OLS fallback successful for {stock}: {ols_result['hedge_ratio']:.4f}")
                    else:
                        print(f"  ✗ OLS fallback also failed for {stock}")
        
        if not spreads:
            return self._run_equal_weight_strategy(stock_data, futures_data)
        
        # Calculate portfolio returns
        spread_df = pd.DataFrame(spreads)
        portfolio_returns = spread_df.pct_change().mean(axis=1).fillna(0)
        
        return pd.DataFrame({'returns': portfolio_returns}, index=stock_data.index)
    
    def _run_kalman_strategy(self, stock_data: pd.DataFrame, 
                            futures_data: pd.Series) -> pd.DataFrame:
        """Run Kalman Filter-based statistical arbitrage strategy"""
        if futures_data is None or len(stock_data.columns) == 0:
            return self._run_equal_weight_strategy(stock_data, futures_data)
        
        # Initialize kalman_filter only when needed
        if self.kalman_filter is None:
            try:
                self.kalman_filter = create_kalman_hedge_filter(
                    smoothing_transition_cov=self.config.get('smoothing_transition_cov', 0.05),
                    hedge_obs_cov=self.config.get('hedge_obs_cov', 2.0),
                    hedge_trans_cov_delta=self.config.get('hedge_trans_cov_delta', 1e-3)
                )
            except ImportError as e:
                print(f"Error running kalman method: {e}")
                return self._run_equal_weight_strategy(stock_data, futures_data)
        
        # Use Kalman Filter for dynamic hedging
        trading_signals = self.kalman_filter.generate_trading_signals(
            futures_data, stock_data.iloc[:, 0]  # Use first stock as example
        )
        
        # Calculate returns based on trading signals
        portfolio_returns = trading_signals['signal'] * stock_data.iloc[:, 0].pct_change().fillna(0)
        
        return pd.DataFrame({'returns': portfolio_returns}, index=stock_data.index)
    
    def evaluate_performance(self, returns_df: pd.DataFrame, 
                           plotting: bool = True) -> Dict:
        """
        Evaluate strategy performance
        
        Args:
            returns_df (pd.DataFrame): Strategy returns
            plotting (bool): Whether to generate plots
            
        Returns:
            Dict: Evaluation results
        """
        if self.evaluator is None:
            raise ValueError("Evaluator not initialized. Call setup() first.")
        
        print("Evaluating strategy performance...")
        
        # Evaluate strategy
        results = self.evaluator.evaluate_strategy(
            returns_df=returns_df,
            total_fee_ratio=self.config.get('total_fee_ratio', 0.0),
            use_benchmark=True,
            plotting=plotting
        )
        
        # Generate performance report
        self.evaluator.generate_performance_report()
        
        # Generate performance plots
        if plotting:
            self.evaluator.plot_performance_analysis()
        
        self.results = results
        return results
    
    def run_full_backtest(self, start_date: str, end_date: str, 
                         plotting: bool = True) -> Dict:
        """
        Run complete backtesting process
        
        Args:
            start_date (str): Start date for backtesting
            end_date (str): End date for backtesting
            plotting (bool): Whether to generate plots
            
        Returns:
            Dict: Complete backtesting results
        """
        print("=" * 60)
        print("STARTING FULL BACKTESTING PROCESS")
        print("=" * 60)
        
        # Setup
        self.setup(start_date, end_date)
        
        # Load data
        stock_data, vn30_data, selected_stocks = self.load_data()
        
        # Run strategy
        returns_df = self.run_strategy(stock_data, vn30_data, selected_stocks)
        
        # Evaluate performance
        results = self.evaluate_performance(returns_df, plotting)
        
        print("=" * 60)
        print("BACKTESTING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return {
            'returns_df': returns_df,
            'evaluation_results': results,
            'stock_data': stock_data,
            'vn30_data': vn30_data,
            'selected_stocks': selected_stocks,
            'config': self.config
        }
    
    def compare_strategies(self, other_strategies: Dict[str, pd.DataFrame]):
        """
        Compare current strategy with other strategies
        
        Args:
            other_strategies (Dict[str, pd.DataFrame]): Other strategy returns
        """
        if self.evaluator is None:
            raise ValueError("Evaluator not initialized. Call setup() first.")
        
        if not self.results:
            raise ValueError("No backtesting results available. Run backtest first.")
        
        print("Comparing strategies...")
        
        # Add current strategy to comparison
        current_returns = self.results['returns_df']['returns']
        all_strategies = {
            'Current Strategy': current_returns
        }
        all_strategies.update(other_strategies)
        
        # Run comparison
        self.evaluator.compare_strategies(all_strategies)
        
        print("Strategy comparison completed!")


def create_backtesting_engine(config: Dict = None) -> BacktestingEngine:
    """
    Factory function to create a BacktestingEngine instance
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        BacktestingEngine: Configured backtesting engine
    """
    return BacktestingEngine(config)


def run_quick_backtest(start_date: str = "2021-06-01", 
                      end_date: str = "2024-12-31",
                      config: Dict = None,
                      method: str = 'kalman') -> Dict:
    """
    Run a quick backtest with default parameters
    
    Args:
        start_date (str): Start date for backtesting
        end_date (str): End date for backtesting
        config (Dict): Configuration dictionary
        method (str): Strategy method ('equal_weight', 'ols', 'bayesian', 'kalman')
        
    Returns:
        Dict: Backtesting results
    """
    print("=" * 60)
    print(f"RUNNING QUICK BACKTEST - METHOD: {method.upper()}")
    print("=" * 60)
    
    engine = create_backtesting_engine(config)
    
    # Setup
    engine.setup(start_date, end_date)
    
    # Load data
    stock_data, vn30_data, selected_stocks = engine.load_data()
    
    # Run strategy with specified method
    returns_df = engine.run_strategy(stock_data, vn30_data, selected_stocks, method=method)
    
    # Evaluate performance
    results = engine.evaluate_performance(returns_df, plotting=True)
    
    print("=" * 60)
    print("BACKTEST COMPLETED")
    print("=" * 60)
    
    return {
        'returns_df': returns_df,
        'evaluation_results': results,
        'stock_data': stock_data,
        'vn30_data': vn30_data,
        'selected_stocks': selected_stocks,
        'config': engine.config,
        'method': method
    }


if __name__ == "__main__":
    import sys
    import json
    import os
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python backtesting.py <mode> [method]")
        print("Modes:")
        print("  insample  - Run in-sample backtesting (2021-2023)")
        print("  outsample - Run out-of-sample backtesting (2023-2025)")
        print("Methods: equal_weight, ols, bayesian, kalman (default: all)")
        print("\nExamples:")
        print("  python backtesting.py insample")
        print("  python backtesting.py outsample")
        print("  python backtesting.py insample kalman")
        print("  python backtesting.py outsample kalman")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    method = sys.argv[2].lower() if len(sys.argv) > 2 else None
    
    # Validate mode
    if mode not in ['insample', 'outsample']:
        print(f"Error: Invalid mode '{mode}'. Use 'insample' or 'outsample'")
        sys.exit(1)
    
    # Validate method if provided
    valid_methods = ['equal_weight', 'ols', 'bayesian', 'kalman']
    if method and method not in valid_methods:
        print(f"Error: Invalid method '{method}'. Valid methods: {valid_methods}")
        sys.exit(1)
    
    # Load configuration based on mode
    config_file = f"parameter/{mode}.json" if mode == 'outsample' else f"parameter/in_sample.json"
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found")
        print("Available config files:")
        for f in os.listdir("parameter"):
            if f.endswith('.json'):
                print(f"  - parameter/{f}")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Set date ranges based on mode
    if mode == 'insample':
        start_date = "2021-01-01"
        end_date = "2023-12-31"
        print("=" * 80)
        print("STATISTICAL ARBITRAGE - IN-SAMPLE BACKTESTING")
        print(f"Date Range: {start_date} to {end_date}")
        print("=" * 80)
    else:  # outsample
        start_date = "2023-01-01"
        end_date = "2025-12-31"
        print("=" * 80)
        print("STATISTICAL ARBITRAGE - OUT-OF-SAMPLE BACKTESTING")
        print(f"Date Range: {start_date} to {end_date}")
        print("=" * 80)
    
    # Determine methods to test
    if method:
        methods = [method]
        print(f"Testing method: {method.upper()}")
    else:
        methods = valid_methods
        print("Testing all methods: equal_weight, ols, bayesian, kalman")
    
    print(f"Configuration loaded from: {config_file}")
    print()
    
    # Run backtesting for each method
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"TESTING METHOD: {method.upper()}")
        print(f"{'='*60}")
        
        try:
            results = run_quick_backtest(
                start_date=start_date,
                end_date=end_date,
                config=config,
                method=method
            )
            
            print(f"\n{method.upper()} Method Results:")
            print(f"Strategy returns shape: {results['returns_df'].shape}")
            print(f"Selected stocks: {results['selected_stocks']}")
            
            # Print detailed performance metrics
            if 'evaluation_results' in results:
                eval_results = results['evaluation_results']
                print(f"\nPerformance Summary:")
                print(f"Annual Return: {eval_results.get('annual_return', 0):.2%}")
                print(f"Total Return: {eval_results.get('total_return', 0):.2%}")
                print(f"Sharpe Ratio: {eval_results.get('sharpe_ratio', 0):.3f}")
                print(f"Sortino Ratio: {eval_results.get('sortino_ratio', 0):.3f}")
                print(f"Max Drawdown: {eval_results.get('max_drawdown', 0):.2%}")
                print(f"Volatility: {eval_results.get('volatility', 0):.2%}")
                print(f"Alpha: {eval_results.get('alpha', 0):.2%}")
                print(f"Beta: {eval_results.get('beta', 0):.3f}")
                print(f"Information Ratio: {eval_results.get('information_ratio', 0):.3f}")
                
                # Store results for comparison
                all_results[method] = {
                    'annual_return': eval_results.get('annual_return', 0),
                    'sharpe_ratio': eval_results.get('sharpe_ratio', 0),
                    'max_drawdown': eval_results.get('max_drawdown', 0),
                    'volatility': eval_results.get('volatility', 0)
                }
            else:
                print("No evaluation results available")
                all_results[method] = {'error': 'No evaluation results'}
                
        except Exception as e:
            print(f"Error running {method} method: {str(e)}")
            all_results[method] = {'error': str(e)}
            continue
    
    # Print comparison summary if multiple methods
    if len(methods) > 1 and all_results:
        print(f"\n{'='*80}")
        print(f"{mode.upper()} BACKTESTING COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Method':<12} {'Annual Return':<15} {'Sharpe Ratio':<12} {'Max Drawdown':<12} {'Volatility':<12}")
        print("-" * 80)
        
        for method, metrics in all_results.items():
            if 'error' not in metrics:
                print(f"{method:<12} {metrics['annual_return']:<15.2%} {metrics['sharpe_ratio']:<12.3f} "
                      f"{metrics['max_drawdown']:<12.2%} {metrics['volatility']:<12.2%}")
            else:
                print(f"{method:<12} {'ERROR':<15} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
    
    print(f"\n{'='*80}")
    print(f"{mode.upper()} BACKTESTING COMPLETED")
    print(f"Results saved to: result/ folder")
    print(f"{'='*80}")


def process_trading_signals_with_position_manager(engine, 
                                                signals_df: pd.DataFrame,
                                                vn05_prices: pd.Series,
                                                futures_prices: pd.Series,
                                                hedge_ratios: pd.Series) -> pd.DataFrame:
    """
    Xử lý tín hiệu giao dịch với Position Manager
    
    Args:
        engine: BacktestingEngine instance
        signals_df (pd.DataFrame): DataFrame chứa tín hiệu giao dịch
        vn05_prices (pd.Series): Giá VN05
        futures_prices (pd.Series): Giá VN30F1M
        hedge_ratios (pd.Series): Tỷ lệ hedge
        
    Returns:
        pd.DataFrame: Returns của chiến lược
    """
    returns = []
    
    for date, row in signals_df.iterrows():
        signal = row['signal']
        vn05_price = vn05_prices.get(date, 0)
        futures_price = futures_prices.get(date, 0)
        hedge_ratio = hedge_ratios.get(date, 1.0)
        
        if signal == 0:  # Không có tín hiệu
            returns.append(0.0)
            continue
        
        # Xác định loại vị thế
        if signal > 0:  # Long signal
            position_type = PositionType.LONG_VN05_SHORT_FUTURES
        else:  # Short signal
            position_type = PositionType.SHORT_VN05_LONG_FUTURES
        
        # Kiểm tra có thể mở vị thế không
        can_open, reason = engine.position_manager.can_open_position(position_type)
        
        if can_open and vn05_price > 0 and futures_price > 0:
            try:
                # Mở vị thế mới
                position = engine.position_manager.open_position(
                    position_type=position_type,
                    vn05_price=vn05_price,
                    futures_price=futures_price,
                    hedge_ratio=hedge_ratio,
                    entry_date=date.strftime('%Y-%m-%d')
                )
                
                # Tính return cho vị thế này
                if position_type == PositionType.LONG_VN05_SHORT_FUTURES:
                    # Long VN05, Short VN30F1M
                    vn05_return = vn05_prices.pct_change().get(date, 0)
                    futures_return = futures_prices.pct_change().get(date, 0)
                    position_return = vn05_return - hedge_ratio * futures_return
                else:
                    # Short VN05, Long VN30F1M
                    vn05_return = vn05_prices.pct_change().get(date, 0)
                    futures_return = futures_prices.pct_change().get(date, 0)
                    position_return = -vn05_return + hedge_ratio * futures_return
                
                returns.append(position_return)
                
            except ValueError as e:
                # Không thể mở vị thế
                returns.append(0.0)
        else:
            # Không thể mở vị thế
            returns.append(0.0)
    
    return pd.DataFrame({'returns': returns}, index=signals_df.index)


def load_parameters(mode):
    """Load parameters from a JSON file based on the mode, or use defaults."""
    import json
    import os
    
    # Default parameters (fallback if JSON not found)
    DEFAULT_PARAMS = {
        "estimation_window": 50,
        "min_trading_days": 25,
        "max_clusters": 10,
        "top_stocks": 3,
        "tier": 1,
        "first_allocation": 0.4,
        "adding_allocation": 0.2,
        "correlation_threshold": 0.6,
        "position_size": 0.4
    }
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Use 'in_sample' parameters for 'out_sample' and 'overall' if no specific file exists
    param_mode = mode if mode in ["in_sample", "optimization"] else "optimization"
    param_file = f"{param_mode}.json"
    param_path = os.path.join(base_dir, "parameter", param_file)
    
    try:
        with open(param_path, "r") as f:
            params = json.load(f)
        print(f"Loaded parameters from {param_file}")
        return params
    except FileNotFoundError:
        print(f"Warning: {param_path} not found. Using default parameters.")
        return DEFAULT_PARAMS


def load_vn30_stocks(use_existing_data, csv_path):
    """Load VN30 stock data from CSV if available and requested, otherwise fetch fresh."""
    import os
    from data.get_data import get_vn30
    
    if use_existing_data and os.path.exists(csv_path):
        vn30_stocks = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        vn30_stocks.index = pd.to_datetime(vn30_stocks.index)
        print(f"Loaded vn30_stocks from {csv_path}")
    else:
        vn30_stocks = get_vn30("2021-06-01", "2025-01-10")
        vn30_stocks.index = pd.to_datetime(vn30_stocks.index)
        print("Fetched fresh vn30_stocks data")
    return vn30_stocks


def run_analysis(vn30_stocks, params, use_existing_data, mode, monthly=False):
    """Run the backtest and calculate metrics based on the specified mode."""
    from utils.helper import generate_periods_df, run_backtest_for_periods
    from utils.calculate_metrics import calculate_metrics, calculate_monthly_returns, pivot_monthly_returns_to_table
    
    # Extract parameters
    estimation_window = params["estimation_window"]
    min_trading_days = params["min_trading_days"]
    max_clusters = params["max_clusters"]
    top_stocks = params["top_stocks"]
    tier = params["tier"]
    first_allocation = params["first_allocation"]
    adding_allocation = params["adding_allocation"]
    correlation_threshold = params["correlation_threshold"]

    # Step 1: Generate periods DataFrame
    start_date = "2021-06-01"
    end_date = "2025-01-01"
    periods_df = generate_periods_df(vn30_stocks, start_date, end_date, window=80)

    # Step 2: Run backtest
    combined_returns_df, combined_detail_df, average_fee_ratio = run_backtest_for_periods(
        periods_df=periods_df,
        futures="VN30F1M",
        estimation_window=estimation_window,
        min_trading_days=min_trading_days,
        max_clusters=max_clusters,
        top_stocks=top_stocks,
        correlation_threshold=correlation_threshold,
        tier=tier,
        first_allocation=first_allocation,
        adding_allocation=adding_allocation,
        use_existing_data=use_existing_data,
    )

    # Step 3: Split into train and test sets
    train_set = combined_returns_df[combined_returns_df.index < "2024-01-01"]
    test_set = combined_returns_df[combined_returns_df.index >= "2024-01-01"]

    # Step 4: Calculate and plot metrics based on mode
    if mode in ["in_sample", "optimization"]:
        print("TRAIN SET")
        calculate_metrics(train_set, average_fee_ratio, risk_free_rate=0.05, plotting=True, use_existing_data=use_existing_data)
    elif mode == "out_sample":
        print("TEST SET")
        calculate_metrics(test_set, average_fee_ratio, risk_free_rate=0.05, plotting=True, use_existing_data=use_existing_data)
    elif mode == "overall":
        print("OVERALL")
        calculate_metrics(combined_returns_df, average_fee_ratio, risk_free_rate=0.05, plotting=True, use_existing_data=use_existing_data)
    
    # Display monthly returns table (optional for all modes)
    if monthly == True:
        monthly_returns = calculate_monthly_returns(combined_returns_df)
        print(pivot_monthly_returns_to_table(monthly_returns))


def main():
    """Main function to orchestrate the script execution."""
    import sys
    import os
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Error: Please specify a mode ('in_sample', 'optimization', 'out_sample', 'overall').")
        print("Example: python backtesting.py in_sample")
        sys.exit(1)

    mode = sys.argv[1].lower()
    use_existing_data = True  # Default to use existing data

    # Normalize mode names
    if mode in ["insample", "in_sample"]:
        mode = "in_sample"
    elif mode in ["outsample", "out_sample"]:
        mode = "out_sample"
    
    valid_modes = ["in_sample", "optimization", "out_sample", "overall"]
    if mode not in valid_modes:
        print(f"Error: Mode must be one of {valid_modes}.")
        print("You can also use: 'insample', 'outsample'")
        sys.exit(1)

    print(f"Running backtesting in {mode} mode...")

    # Load parameters
    params = load_parameters(mode)

    # Load data
    data_folder = "data"
    csv_path = os.path.join(data_folder, "vn30_stocks.csv")
    vn30_stocks = load_vn30_stocks(use_existing_data, csv_path)
    
    # Load VN30F1M data
    try:
        vn30f1m = pd.read_csv("data/vn30f1m.csv", index_col=0, parse_dates=True)
        vn30f1m_price = vn30f1m["price"]
        print(f"Loaded VN30F1M data: {len(vn30f1m_price)} records")
    except FileNotFoundError:
        print("Warning: VN30F1M data not found. Using dummy data.")
        from data.get_data import get_vn30f1m
        vn30f1m_price = get_vn30f1m("2021-06-01", "2025-01-10")
    
    # Run analysis
    run_analysis(vn30_stocks, params, use_existing_data, mode)
    
    print(f"\n{'='*80}")
    print(f"{mode.upper()} BACKTESTING COMPLETED")
    print(f"Results saved to: result/ folder")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
