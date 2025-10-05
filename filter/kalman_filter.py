"""
Production Kalman Filter module for statistical arbitrage
Based on advanced production code with price smoothing and dynamic hedge ratios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from pykalman import KalmanFilter
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False
    print("Warning: pykalman not available. Install with: pip install pykalman")

try:
    import pymc3 as pm
    import theano.tensor as tt
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("Warning: PyMC3 not available. Install with: pip install pymc3")

class KalmanHedgeFilter:
    """
    Production Kalman Filter with price smoothing and dynamic hedge ratios
    """
    
    def __init__(self, 
                 smoothing_transition_cov: float = 0.05,
                 hedge_obs_cov: float = 2.0,
                 hedge_trans_cov_delta: float = 1e-3,
                 initial_state_mean: float = 0.0,
                 initial_state_cov: float = 1.0):
        """
        Initialize Production Kalman Filter
        
        Args:
            smoothing_transition_cov (float): Transition covariance for price smoothing
            hedge_obs_cov (float): Observation covariance for hedge ratio
            hedge_trans_cov_delta (float): Delta for transition covariance calculation
            initial_state_mean (float): Initial state mean
            initial_state_cov (float): Initial state covariance
        """
        if not PYKALMAN_AVAILABLE:
            raise ImportError("pykalman is required for KalmanHedgeFilter")
            
        self.smoothing_transition_cov = smoothing_transition_cov
        self.hedge_obs_cov = hedge_obs_cov
        self.hedge_trans_cov_delta = hedge_trans_cov_delta
        self.initial_state_mean = initial_state_mean
        self.initial_state_cov = initial_state_cov
        
        # Results storage
        self.smoothed_prices = {}
        self.hedge_ratios = {}
        self.half_lives = {}
        self.spreads = {}
        self.z_scores = {}
    
    def smooth_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Kalman Filter smoothing to prices
        
        Args:
            prices (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Smoothed prices
        """
        if not PYKALMAN_AVAILABLE:
            return prices
        
        def kf_smoother(price_series):
            """Estimate rolling mean using Kalman Filter"""
            kf = KalmanFilter(
                transition_matrices=np.eye(1),
                observation_matrices=np.eye(1),
                initial_state_mean=self.initial_state_mean,
                initial_state_covariance=self.initial_state_cov,
                observation_covariance=1,
                transition_covariance=self.smoothing_transition_cov
            )
            
            state_means, _ = kf.filter(price_series.values)
            return pd.Series(state_means.flatten(), index=price_series.index)
        
        smoothed = prices.apply(kf_smoother)
        self.smoothed_prices = smoothed
        return smoothed
    
    def estimate_hedge_ratio(self, x: pd.Series, y: pd.Series) -> np.ndarray:
        """
        Estimate hedge ratio using Kalman Filter
        
        Args:
            x (pd.Series): Independent variable
            y (pd.Series): Dependent variable
            
        Returns:
            np.ndarray: Hedge ratio estimates [beta, alpha]
        """
        if not PYKALMAN_AVAILABLE:
            # Fallback to simple OLS
            X = np.column_stack([x.values, np.ones(len(x))])
            return np.linalg.lstsq(X, y.values, rcond=None)[0]
        
        # Prepare observation matrix
        delta = self.hedge_trans_cov_delta
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
        
        # Initialize Kalman Filter
        kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=2,
            initial_state_mean=[0, 0],
            initial_state_covariance=np.ones((2, 2)),
            transition_matrices=np.eye(2),
            observation_matrices=obs_mat,
            observation_covariance=self.hedge_obs_cov,
            transition_covariance=trans_cov
        )
        
        # Filter
        state_means, _ = kf.filter(y.values)
        return -state_means  # Return negative for short position
    
    def estimate_half_life(self, spread: pd.Series) -> int:
        """
        Estimate mean reversion half-life
        
        Args:
            spread (pd.Series): Spread series
            
        Returns:
            int: Half-life in periods
        """
        try:
            # Prepare data
            X = spread.shift().iloc[1:].to_frame().assign(const=1)
            y = spread.diff().iloc[1:]
            
            # Remove NaN values
            valid_idx = ~(X.iloc[:, 0].isna() | y.isna())
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]
            
            if len(X_clean) < 5:
                return 1
            
            # OLS regression
            beta = (np.linalg.inv(X_clean.T @ X_clean) @ X_clean.T @ y_clean).iloc[0]
            
            # Calculate half-life
            if beta >= 0:
                return 1  # No mean reversion
            
            halflife = int(round(-np.log(2) / beta, 0))
            return max(halflife, 1)
            
        except Exception:
            return 1
    
    def calculate_spread_with_dynamic_window(self, 
                                           y: pd.Series, 
                                           x: pd.Series,
                                           max_window: int = 252) -> pd.DataFrame:
        """
        Calculate spread with dynamic window based on half-life
        
        Args:
            y (pd.Series): Dependent variable
            x (pd.Series): Independent variable
            max_window (int): Maximum window size
            
        Returns:
            pd.DataFrame: Spread analysis results
        """
        # Smooth prices first
        y_smooth = self.smooth_prices(pd.DataFrame({'y': y}))['y']
        x_smooth = self.smooth_prices(pd.DataFrame({'x': x}))['x']
        
        # Estimate hedge ratio
        hedge_ratio = self.estimate_hedge_ratio(x_smooth, y_smooth)
        
        # Calculate spread
        spread = y + x * hedge_ratio[:, 0]  # Use beta (first element)
        
        # Estimate half-life
        half_life = self.estimate_half_life(spread)
        
        # Calculate z-score with dynamic window
        window = min(2 * half_life, max_window)
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        z_score = (spread - rolling_mean) / rolling_std
        
        # Store results
        self.hedge_ratios[(y.name, x.name)] = hedge_ratio
        self.half_lives[(y.name, x.name)] = half_life
        self.spreads[(y.name, x.name)] = spread
        
        return pd.DataFrame({
            'spread': spread,
            'z_score': z_score,
            'hedge_ratio': hedge_ratio[:, 0],
            'half_life': half_life,
            'window': window
        }, index=y.index)
    
    def generate_advanced_signals(self, 
                                 spread_data: pd.DataFrame,
                                 entry_threshold: float = 2.0,
                                 exit_threshold: float = 0.5) -> pd.DataFrame:
        """
        Generate advanced trading signals
        
        Args:
            spread_data (pd.DataFrame): Spread analysis data
            entry_threshold (float): Entry threshold
            exit_threshold (float): Exit threshold
            
        Returns:
            pd.DataFrame: Trading signals
        """
        z_score = spread_data['z_score']
        
        # Entry signals
        entry = z_score.abs() > entry_threshold
        entry_signals = ((entry.shift() != entry)
                        .mul(np.sign(z_score))
                        .fillna(0)
                        .astype(int)
                        .sub(2))
        
        # Exit signals
        exit_signals = (np.sign(z_score.shift().fillna(method='bfill'))
                       != np.sign(z_score)).astype(int) - 1
        
        # Combine signals
        signals = pd.Series(0, index=z_score.index)
        signals[entry_signals == -1] = 1   # Long signal
        signals[entry_signals == 1] = -1   # Short signal
        signals[exit_signals == 0] = 0     # Exit signal
        
        return pd.DataFrame({
            'z_score': z_score,
            'signal': signals,
            'entry_signals': entry_signals,
            'exit_signals': exit_signals,
            'spread': spread_data['spread'],
            'hedge_ratio': spread_data['hedge_ratio']
        })
    
    def process_pairs_batch(self, 
                           candidates: pd.DataFrame,
                           prices: pd.DataFrame,
                           test_periods: List[pd.Timestamp]) -> List[pd.DataFrame]:
        """
        Process multiple pairs in batch
        
        Args:
            candidates (pd.DataFrame): Pair candidates
            prices (pd.DataFrame): Price data
            test_periods (List): Test periods
            
        Returns:
            List[pd.DataFrame]: Processed pair data
        """
        results = []
        
        for period in test_periods:
            period_candidates = candidates[candidates['test_end'] == period]
            
            for _, row in period_candidates.iterrows():
                y_name, x_name = row['y'], row['x']
                
                if y_name not in prices.columns or x_name not in prices.columns:
                    continue
                
                try:
                    # Calculate spread with dynamic window
                    spread_data = self.calculate_spread_with_dynamic_window(
                        prices[y_name], prices[x_name]
                    )
                    
                    # Generate signals
                    signals = self.generate_advanced_signals(spread_data)
                    
                    # Add metadata
                    signals['y'] = y_name
                    signals['x'] = x_name
                    signals['period'] = period
                    
                    results.append(signals)
                    
                except Exception as e:
                    print(f"Error processing {y_name}-{x_name}: {e}")
                    continue
        
        return results
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics
        
        Returns:
            Dict: Performance metrics
        """
        metrics = {
            'total_pairs': len(self.hedge_ratios),
            'avg_half_life': np.mean(list(self.half_lives.values())),
            'median_half_life': np.median(list(self.half_lives.values())),
            'hedge_ratios_stats': {
                'mean': np.mean([hr[:, 0].mean() for hr in self.hedge_ratios.values()]),
                'std': np.std([hr[:, 0].mean() for hr in self.hedge_ratios.values()])
            }
        }
        
        return metrics
    
    def generate_trading_signals(self, futures_data: pd.Series, stock_data: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals using Kalman filter
        
        Args:
            futures_data (pd.Series): Futures price data
            stock_data (pd.Series): Stock price data
            
        Returns:
            pd.DataFrame: Trading signals and hedge ratios
        """
        try:
            # Align data
            aligned_data = pd.concat([futures_data, stock_data], axis=1, join='inner').dropna()
            if len(aligned_data) < 10:
                # Return empty signals if insufficient data
                return pd.DataFrame({
                    'signal': [0] * len(futures_data),
                    'hedge_ratio': [1.0] * len(futures_data)
                }, index=futures_data.index)
            
            futures_clean = aligned_data.iloc[:, 0]
            stock_clean = aligned_data.iloc[:, 1]
            
            # Calculate rolling hedge ratio using simple linear regression
            hedge_ratios = []
            signals = []
            
            window = min(30, len(aligned_data) // 3)
            
            for i in range(len(aligned_data)):
                if i < window:
                    hedge_ratio = 1.0
                    signal = 0
                else:
                    # Use rolling window for hedge ratio calculation
                    y_window = futures_clean.iloc[i-window:i]
                    x_window = stock_clean.iloc[i-window:i]
                    
                    # Simple linear regression: y = alpha + beta * x
                    beta = np.cov(y_window, x_window)[0, 1] / np.var(x_window) if np.var(x_window) > 0 else 1.0
                    alpha = y_window.mean() - beta * x_window.mean()
                    
                    # Calculate spread
                    spread = futures_clean.iloc[i] - (alpha + beta * stock_clean.iloc[i])
                    
                    # Simple mean reversion signal
                    spread_mean = np.mean([futures_clean.iloc[j] - (alpha + beta * stock_clean.iloc[j]) 
                                        for j in range(max(0, i-20), i)])
                    spread_std = np.std([futures_clean.iloc[j] - (alpha + beta * stock_clean.iloc[j]) 
                                       for j in range(max(0, i-20), i)])
                    
                    if spread_std > 0:
                        z_score = (spread - spread_mean) / spread_std
                        if z_score > 2:
                            signal = -1  # Short spread (sell futures, buy stock)
                        elif z_score < -2:
                            signal = 1   # Long spread (buy futures, sell stock)
                        else:
                            signal = 0
                    else:
                        signal = 0
                    
                    hedge_ratio = beta
                
                hedge_ratios.append(hedge_ratio)
                signals.append(signal)
            
            # Create result DataFrame
            result = pd.DataFrame({
                'signal': signals,
                'hedge_ratio': hedge_ratios
            }, index=aligned_data.index)
            
            # Reindex to match original futures_data index
            result = result.reindex(futures_data.index, fill_value=0)
            
            return result
            
        except Exception as e:
            print(f"Error generating trading signals: {e}")
            # Return empty signals on error
            return pd.DataFrame({
                'signal': [0] * len(futures_data),
                'hedge_ratio': [1.0] * len(futures_data)
            }, index=futures_data.index)


def create_kalman_hedge_filter(smoothing_transition_cov: float = 0.05,
                              hedge_obs_cov: float = 2.0,
                              hedge_trans_cov_delta: float = 1e-3) -> KalmanHedgeFilter:
    """
    Factory function to create Kalman Hedge Filter
    
    Args:
        smoothing_transition_cov (float): Transition covariance for smoothing
        hedge_obs_cov (float): Observation covariance for hedge ratio
        hedge_trans_cov_delta (float): Delta for transition covariance
        
    Returns:
        KalmanHedgeFilter: Configured filter
    """
    return KalmanHedgeFilter(
        smoothing_transition_cov=smoothing_transition_cov,
        hedge_obs_cov=hedge_obs_cov,
        hedge_trans_cov_delta=hedge_trans_cov_delta
    )


if __name__ == "__main__":
    # Example usage
    if PYKALMAN_AVAILABLE:
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2021-01-01', periods=100, freq='D')
        
        # Generate correlated data
        x = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates, name='X')
        y = pd.Series(1.5 * x + np.random.randn(100) * 0.1, index=dates, name='Y')
        
        # Create kalman filter
        kf = create_kalman_hedge_filter()
        
        # Process single pair
        spread_data = kf.calculate_spread_with_dynamic_window(y, x)
        signals = kf.generate_advanced_signals(spread_data)
        
        print("Kalman Hedge Filter Results:")
        print(f"Half-life: {kf.half_lives[('Y', 'X')]}")
        print(f"Average hedge ratio: {spread_data['hedge_ratio'].mean():.4f}")
        print(f"Signal distribution:")
        print(signals['signal'].value_counts())
        
        # Get performance metrics
        metrics = kf.get_performance_metrics()
        print(f"\nPerformance Metrics:")
        print(f"Average half-life: {metrics['avg_half_life']:.2f}")
        print(f"Hedge ratio mean: {metrics['hedge_ratios_stats']['mean']:.4f}")
        
    else:
        print("pykalman not available. Install with: pip install pykalman")