"""
Production OLS estimation module for statistical arbitrage
Enhanced with rolling OLS, half-life estimation, and advanced features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings('ignore')

class OLSEstimator:
    """
    Production OLS estimator with rolling regression and half-life estimation
    """
    
    def __init__(self, min_periods: int = 30, confidence_level: float = 0.05,
                 rolling_window: int = 20, max_window: int = 252):
        """
        Initialize OLS estimator
        
        Args:
            min_periods (int): Minimum periods required for estimation
            confidence_level (float): Confidence level for cointegration tests
            rolling_window (int): Default rolling window for rolling OLS
            max_window (int): Maximum window size for calculations
        """
        self.min_periods = min_periods
        self.confidence_level = confidence_level
        self.rolling_window = rolling_window
        self.max_window = max_window
        self.estimation_results = {}
    
    def estimate_half_life(self, spread: pd.Series) -> int:
        """
        Estimate mean reversion half-life using AR(1) model
        
        Args:
            spread (pd.Series): Spread series
            
        Returns:
            int: Half-life in periods
        """
        try:
            # Prepare data for AR(1) regression: spread_t = alpha + beta * spread_{t-1} + epsilon
            X = spread.shift().iloc[1:].to_frame().assign(const=1)
            y = spread.diff().iloc[1:]
            
            # Remove NaN values
            valid_idx = ~(X.iloc[:, 0].isna() | y.isna())
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]
            
            if len(X_clean) < 5:
                return 1
            
            # OLS regression: spread_t - spread_{t-1} = alpha + beta * spread_{t-1}
            # This is equivalent to: spread_t = alpha + (1 + beta) * spread_{t-1}
            beta = (np.linalg.inv(X_clean.T @ X_clean) @ X_clean.T @ y_clean).iloc[0]
            
            # Calculate half-life: t_half = -ln(2) / ln(1 + beta)
            # For mean reversion, beta should be negative
            if beta >= 0:
                return 1  # No mean reversion detected
            
            halflife = int(round(-np.log(2) / beta, 0))
            return max(halflife, 1)
            
        except Exception as e:
            print(f"Error estimating half-life: {e}")
            return 1
    
    def calculate_hedge_ratio_rolling(self, y: pd.Series, x: pd.Series, 
                                    window: int = None) -> pd.DataFrame:
        """
        Calculate rolling hedge ratio between two assets
        
        Args:
            y (pd.Series): Dependent variable (e.g., futures price)
            x (pd.Series): Independent variable (e.g., spot price)
            window (int): Rolling window size (default: self.rolling_window)
            
        Returns:
            pd.DataFrame: Rolling hedge ratio results
        """
        if window is None:
            window = self.rolling_window
        
        # Align data
        aligned_data = pd.concat([y, x], axis=1, join='inner').dropna()
        y_clean = aligned_data.iloc[:, 0]
        x_clean = aligned_data.iloc[:, 1]
        
        if len(aligned_data) < window:
            return pd.DataFrame()
        
        # Rolling regression
        hedge_ratios = []
        r_squareds = []
        p_values = []
        residuals = []
        
        for i in range(window, len(aligned_data) + 1):
            y_window = y_clean.iloc[i-window:i]
            x_window = x_clean.iloc[i-window:i]
            
            try:
                # OLS regression: y = alpha + beta * x
                X = np.column_stack([x_window.values, np.ones(len(x_window))])
                model = OLS(y_window.values, X).fit()
                
                hedge_ratio = model.params[0]  # beta coefficient
                r_squared = model.rsquared
                p_value = model.pvalues[0]
                residual = y_window.iloc[-1] - (model.params[1] + model.params[0] * x_window.iloc[-1])
                
                hedge_ratios.append(hedge_ratio)
                r_squareds.append(r_squared)
                p_values.append(p_value)
                residuals.append(residual)
                
            except Exception:
                hedge_ratios.append(np.nan)
                r_squareds.append(np.nan)
                p_values.append(np.nan)
                residuals.append(np.nan)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'hedge_ratio': hedge_ratios,
            'r_squared': r_squareds,
            'p_value': p_values,
            'residual': residuals
        }, index=aligned_data.index[window-1:])
        
        return results
    
    def calculate_spread_with_dynamic_window(self, y: pd.Series, x: pd.Series,
                                           max_window: int = None) -> pd.DataFrame:
        """
        Calculate spread with dynamic window based on half-life
        
        Args:
            y (pd.Series): Dependent variable
            x (pd.Series): Independent variable
            max_window (int): Maximum window size
            
        Returns:
            pd.DataFrame: Spread analysis results
        """
        if max_window is None:
            max_window = self.max_window
        
        # Calculate rolling hedge ratio
        rolling_results = self.calculate_hedge_ratio_rolling(y, x)
        
        if rolling_results.empty:
            return pd.DataFrame()
        
        # Calculate spread using latest hedge ratio
        latest_hedge_ratio = rolling_results['hedge_ratio'].iloc[-1]
        if np.isnan(latest_hedge_ratio):
            latest_hedge_ratio = 1.0  # Fallback
        
        spread = y - latest_hedge_ratio * x
        
        # Estimate half-life
        half_life = self.estimate_half_life(spread)
        
        # Calculate z-score with dynamic window
        window = min(2 * half_life, max_window)
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        z_score = (spread - rolling_mean) / rolling_std
        
        return pd.DataFrame({
            'spread': spread,
            'z_score': z_score,
            'hedge_ratio': latest_hedge_ratio,
            'half_life': half_life,
            'window': window,
            'rolling_r_squared': rolling_results['r_squared'],
            'rolling_p_value': rolling_results['p_value']
        }, index=y.index)
    
    def calculate_hedge_ratio(self, y: pd.Series, x: pd.Series, 
                            method: str = 'ols') -> Dict:
        """
        Calculate hedge ratio between two assets
        
        Args:
            y (pd.Series): Dependent variable (e.g., futures price)
            x (pd.Series): Independent variable (e.g., spot price)
            method (str): Estimation method ('ols', 'cointegration', 'rolling')
            
        Returns:
            Dict: Estimation results including hedge ratio, statistics
        """
        # Align data
        aligned_data = pd.concat([y, x], axis=1, join='inner').dropna()
        
        if len(aligned_data) < self.min_periods:
            return {
                'hedge_ratio': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'cointegration_pvalue': np.nan,
                'adf_pvalue': np.nan,
                'half_life': 1,
                'method': method,
                'error': 'Insufficient data'
            }
        
        y_clean = aligned_data.iloc[:, 0]
        x_clean = aligned_data.iloc[:, 1]
        
        try:
            if method == 'rolling':
                # Use rolling OLS
                rolling_results = self.calculate_hedge_ratio_rolling(y_clean, x_clean)
                if rolling_results.empty:
                    return self._fallback_estimation(y_clean, x_clean, method)
                
                # Use latest rolling results
                latest_idx = rolling_results.index[-1]
                hedge_ratio = rolling_results.loc[latest_idx, 'hedge_ratio']
                r_squared = rolling_results.loc[latest_idx, 'r_squared']
                p_value = rolling_results.loc[latest_idx, 'p_value']
                
            else:
                # Standard OLS
                X = np.column_stack([x_clean.values, np.ones(len(x_clean))])
                model = OLS(y_clean.values, X).fit()
                
                hedge_ratio = model.params[0]
                r_squared = model.rsquared
                p_value = model.pvalues[0]
            
            # Calculate spread for additional tests
            spread = y_clean - hedge_ratio * x_clean
            
            # Cointegration test
            try:
                coint_stat, coint_pvalue, _, _, _, _ = coint(y_clean, x_clean)
            except:
                coint_pvalue = np.nan
            
            # ADF test on residuals
            try:
                maxlag = min(1, len(spread) // 2 - 1) if len(spread) > 4 else 0
                if maxlag > 0:
                    adf_pvalue = adfuller(spread, maxlag=maxlag, autolag="AIC")[1]
                else:
                    adf_pvalue = 1.0
            except:
                adf_pvalue = np.nan
            
            # Estimate half-life
            half_life = self.estimate_half_life(spread)
            
            result = {
                'hedge_ratio': hedge_ratio,
                'r_squared': r_squared,
                'p_value': p_value,
                'cointegration_pvalue': coint_pvalue,
                'adf_pvalue': adf_pvalue,
                'half_life': half_life,
                'method': method,
                'n_observations': len(aligned_data),
                'is_cointegrated': coint_pvalue < self.confidence_level if not np.isnan(coint_pvalue) else False,
                'error': None
            }
            
            # Store results
            self.estimation_results[(y.name, x.name)] = result
            
            return result
            
        except Exception as e:
            return self._fallback_estimation(y_clean, x_clean, method, str(e))
    
    def _fallback_estimation(self, y: pd.Series, x: pd.Series, 
                           method: str, error: str = None) -> Dict:
        """
        Fallback estimation when main method fails
        
        Args:
            y (pd.Series): Dependent variable
            x (pd.Series): Independent variable
            method (str): Original method attempted
            error (str): Error message
            
        Returns:
            Dict: Fallback results
        """
        try:
            # Simple correlation-based hedge ratio
            correlation = y.corr(x)
            hedge_ratio = correlation * y.std() / x.std() if x.std() > 0 else 1.0
            
            # Calculate basic statistics
            spread = y - hedge_ratio * x
            r_squared = correlation ** 2
            half_life = self.estimate_half_life(spread)
            
            return {
                'hedge_ratio': hedge_ratio,
                'r_squared': r_squared,
                'p_value': np.nan,
                'cointegration_pvalue': np.nan,
                'adf_pvalue': np.nan,
                'half_life': half_life,
                'method': f'{method}_fallback',
                'n_observations': len(y),
                'error': error or 'Fallback estimation used'
            }
        except:
            return {
                'hedge_ratio': 1.0,
                'r_squared': 0.0,
                'p_value': np.nan,
                'cointegration_pvalue': np.nan,
                'adf_pvalue': np.nan,
                'half_life': 1,
                'method': f'{method}_error',
                'n_observations': len(y),
                'error': error or 'Estimation failed completely'
            }
    
    def generate_trading_signals(self, spread_data: pd.DataFrame,
                               entry_threshold: float = 2.0,
                               exit_threshold: float = 0.5) -> pd.DataFrame:
        """
        Generate trading signals based on spread analysis
        
        Args:
            spread_data (pd.DataFrame): Spread analysis results
            entry_threshold (float): Entry threshold for z-score
            exit_threshold (float): Exit threshold for z-score
            
        Returns:
            pd.DataFrame: Trading signals
        """
        z_score = spread_data['z_score']
        
        # Generate signals
        signals = pd.Series(0, index=z_score.index)
        signals[z_score < -entry_threshold] = 1   # Long signal
        signals[z_score > entry_threshold] = -1   # Short signal
        signals[(z_score > -exit_threshold) & (z_score < exit_threshold)] = 0  # Exit
        
        return pd.DataFrame({
            'z_score': z_score,
            'signal': signals,
            'spread': spread_data['spread'],
            'hedge_ratio': spread_data['hedge_ratio'],
            'half_life': spread_data['half_life']
        })
    
    def process_pairs_batch(self, candidates: pd.DataFrame,
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
                    signals = self.generate_trading_signals(spread_data)
                    
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
        if not self.estimation_results:
            return {'error': 'No estimation results available'}
        
        hedge_ratios = [result['hedge_ratio'] for result in self.estimation_results.values()]
        half_lives = [result['half_life'] for result in self.estimation_results.values()]
        r_squareds = [result['r_squared'] for result in self.estimation_results.values()]
        
        return {
            'total_pairs': len(self.estimation_results),
            'avg_hedge_ratio': np.mean(hedge_ratios),
            'std_hedge_ratio': np.std(hedge_ratios),
            'avg_half_life': np.mean(half_lives),
            'median_half_life': np.median(half_lives),
            'avg_r_squared': np.mean(r_squareds),
            'successful_estimations': len([r for r in self.estimation_results.values() if r['error'] is None])
        }
    
    def calculate_spread(self, y: pd.Series, x: pd.Series, hedge_ratio: float) -> pd.Series:
        """
        Calculate spread between two series using given hedge ratio
        
        Args:
            y (pd.Series): First series (e.g., futures)
            x (pd.Series): Second series (e.g., spot)
            hedge_ratio (float): Hedge ratio to use
            
        Returns:
            pd.Series: Spread series
        """
        # Align data
        aligned_data = pd.concat([y, x], axis=1, join='inner').dropna()
        if len(aligned_data) == 0:
            return pd.Series(dtype=float)
        
        y_clean = aligned_data.iloc[:, 0]
        x_clean = aligned_data.iloc[:, 1]
        
        # Calculate spread: y - hedge_ratio * x
        spread = y_clean - hedge_ratio * x_clean
        
        return spread


def create_ols_estimator(min_periods: int = 30,
                        confidence_level: float = 0.05,
                        rolling_window: int = 20,
                        max_window: int = 252) -> OLSEstimator:
    """
    Factory function to create OLS Estimator
    
    Args:
        min_periods (int): Minimum periods required
        confidence_level (float): Confidence level for tests
        rolling_window (int): Default rolling window
        max_window (int): Maximum window size
        
    Returns:
        OLSEstimator: Configured estimator
    """
    return OLSEstimator(
        min_periods=min_periods,
        confidence_level=confidence_level,
        rolling_window=rolling_window,
        max_window=max_window
    )


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2021-01-01', periods=100, freq='D')
    
    # Generate correlated data
    x = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates, name='X')
    y = pd.Series(1.5 * x + np.random.randn(100) * 0.1, index=dates, name='Y')
    
    # Create OLS estimator
    ols = create_ols_estimator()
    
    # Test different methods
    methods = ['ols', 'rolling']
    
    for method in methods:
        print(f"\n{method.upper()} Method Results:")
        result = ols.calculate_hedge_ratio(y, x, method=method)
        print(f"Hedge ratio: {result['hedge_ratio']:.4f}")
        print(f"R-squared: {result['r_squared']:.4f}")
        print(f"Half-life: {result['half_life']}")
        print(f"ADF p-value: {result['adf_pvalue']:.4f}")
    
    # Test spread calculation with dynamic window
    print(f"\nSpread Analysis with Dynamic Window:")
    spread_data = ols.calculate_spread_with_dynamic_window(y, x)
    print(f"Average z-score: {spread_data['z_score'].mean():.4f}")
    print(f"Z-score std: {spread_data['z_score'].std():.4f}")
    print(f"Half-life: {spread_data['half_life'].iloc[0]}")
    
    # Generate trading signals
    signals = ols.generate_trading_signals(spread_data)
    print(f"\nTrading Signals:")
    print(f"Long signals: {(signals['signal'] == 1).sum()}")
    print(f"Short signals: {(signals['signal'] == -1).sum()}")
    print(f"Neutral signals: {(signals['signal'] == 0).sum()}")
    
    # Get performance metrics
    metrics = ols.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"Average hedge ratio: {metrics['avg_hedge_ratio']:.4f}")
    print(f"Average half-life: {metrics['avg_half_life']:.2f}")
    print(f"Average R-squared: {metrics['avg_r_squared']:.4f}")