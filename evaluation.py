"""
Evaluation module for strategy performance evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from metrics.metric import (
    calculate_metrics, calculate_shapre_and_mdd,
    calculate_monthly_returns, calculate_yearly_returns,
    pivot_monthly_returns_to_table
)
import os
from datetime import datetime

class StrategyEvaluator:
    """
    Class for evaluating strategy performance
    """
    
    def __init__(self, risk_free_rate: float = 0.05, trading_days: int = 252):
        """
        Initialize the strategy evaluator
        
        Args:
            risk_free_rate (float): Annual risk-free rate
            trading_days (int): Number of trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.evaluation_results = {}
    
    def evaluate_strategy(self, returns_df: pd.DataFrame, 
                         total_fee_ratio: float = 0.0,
                         use_benchmark: bool = True,
                         plotting: bool = False) -> Dict:
        """
        Evaluate strategy performance
        
        Args:
            returns_df (pd.DataFrame): Strategy returns with 'returns' column
            total_fee_ratio (float): Total fee ratio for turnover calculation
            use_benchmark (bool): Whether to include benchmark comparison
            plotting (bool): Whether to generate plots
            
        Returns:
            Dict: Evaluation results
        """
        print("Evaluating strategy performance...")
        
        # Calculate comprehensive metrics
        metrics_df = calculate_metrics(
            returns_df=returns_df,
            total_fee_ratio=total_fee_ratio,
            risk_free_rate=self.risk_free_rate,
            trading_day=self.trading_days,
            use_benchmark=use_benchmark,
            plotting=plotting
        )
        
        # Calculate basic metrics (for compatibility)
        annual_return, sharpe_ratio, max_drawdown = calculate_shapre_and_mdd(
            returns_df=returns_df,
            risk_free_rate=self.risk_free_rate,
            trading_day=self.trading_days
        )
        
        # Calculate monthly returns
        monthly_returns = calculate_monthly_returns(returns_df)
        
        # Calculate yearly returns
        yearly_returns = calculate_yearly_returns(returns_df)
        
        # Store results
        self.evaluation_results = {
            'metrics_df': metrics_df,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'monthly_returns': monthly_returns,
            'yearly_returns': yearly_returns,
            'returns_df': returns_df
        }
        
        return self.evaluation_results
    
    def generate_performance_report(self, output_dir: str = "result") -> None:
        """
        Generate comprehensive performance report
        
        Args:
            output_dir (str): Output directory for reports
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_strategy() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to CSV
        metrics_file = os.path.join(output_dir, "strategy_metrics.csv")
        self.evaluation_results['metrics_df'].to_csv(metrics_file)
        print(f"Metrics saved to {metrics_file}")
        
        # Save monthly returns
        monthly_file = os.path.join(output_dir, "monthly_returns.csv")
        self.evaluation_results['monthly_returns'].to_csv(monthly_file, index=False)
        print(f"Monthly returns saved to {monthly_file}")
        
        # Save yearly returns
        yearly_file = os.path.join(output_dir, "yearly_returns.csv")
        self.evaluation_results['yearly_returns'].to_csv(yearly_file, index=False)
        print(f"Yearly returns saved to {yearly_file}")
        
        # Generate monthly returns table
        monthly_table = pivot_monthly_returns_to_table(self.evaluation_results['monthly_returns'])
        table_file = os.path.join(output_dir, "monthly_returns_table.csv")
        monthly_table.to_csv(table_file)
        print(f"Monthly returns table saved to {table_file}")
        
        # Generate summary report
        self._generate_summary_report(output_dir)
    
    def _generate_summary_report(self, output_dir: str) -> None:
        """
        Generate summary performance report
        
        Args:
            output_dir (str): Output directory for reports
        """
        report_file = os.path.join(output_dir, "performance_summary.txt")
        
        with open(report_file, 'w') as f:
            f.write("STRATEGY PERFORMANCE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Risk-free Rate: {self.risk_free_rate:.2%}\n")
            f.write(f"Trading Days per Year: {self.trading_days}\n\n")
            
            # Basic metrics
            f.write("BASIC METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Annual Return: {self.evaluation_results['annual_return']:.2%}\n")
            f.write(f"Sharpe Ratio: {self.evaluation_results['sharpe_ratio']:.3f}\n")
            f.write(f"Maximum Drawdown: {self.evaluation_results['max_drawdown']:.2%}\n\n")
            
            # Data period
            returns_df = self.evaluation_results['returns_df']
            f.write("DATA PERIOD:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Start Date: {returns_df.index.min().strftime('%Y-%m-%d')}\n")
            f.write(f"End Date: {returns_df.index.max().strftime('%Y-%m-%d')}\n")
            f.write(f"Total Days: {len(returns_df)}\n")
            f.write(f"Trading Days: {len(returns_df.dropna())}\n\n")
            
            # Monthly performance summary
            monthly_returns = self.evaluation_results['monthly_returns']
            if not monthly_returns.empty:
                f.write("MONTHLY PERFORMANCE:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Best Month: {monthly_returns['Monthly Return'].max():.2%}\n")
                f.write(f"Worst Month: {monthly_returns['Monthly Return'].min():.2%}\n")
                f.write(f"Average Monthly Return: {monthly_returns['Monthly Return'].mean():.2%}\n")
                f.write(f"Monthly Volatility: {monthly_returns['Monthly Return'].std():.2%}\n")
                f.write(f"Positive Months: {(monthly_returns['Monthly Return'] > 0).sum()}\n")
                f.write(f"Negative Months: {(monthly_returns['Monthly Return'] < 0).sum()}\n\n")
            
            # Yearly performance summary
            yearly_returns = self.evaluation_results['yearly_returns']
            if not yearly_returns.empty:
                f.write("YEARLY PERFORMANCE:\n")
                f.write("-" * 18 + "\n")
                for _, row in yearly_returns.iterrows():
                    f.write(f"{int(row['Year'])}: {row['Yearly Return']:.2%}\n")
        
        print(f"Performance summary saved to {report_file}")
    
    def plot_performance_analysis(self, output_dir: str = "result") -> None:
        """
        Generate performance analysis plots
        
        Args:
            output_dir (str): Output directory for plots
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_strategy() first.")
            return
        
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        returns_df = self.evaluation_results['returns_df']
        
        # 1. Cumulative returns plot
        plt.figure(figsize=(12, 8))
        cumulative_returns = (1 + returns_df['returns']).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2)
        plt.title('Strategy Cumulative Returns', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'cumulative_returns.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Monthly returns bar chart
        monthly_returns = self.evaluation_results['monthly_returns']
        if not monthly_returns.empty:
            plt.figure(figsize=(15, 8))
            monthly_returns['Date'] = pd.to_datetime(monthly_returns[['Year', 'Month']].assign(day=1))
            plt.bar(monthly_returns['Date'], monthly_returns['Monthly Return'] * 100, 
                   alpha=0.7, color='steelblue')
            plt.title('Monthly Returns (%)', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Monthly Return (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'monthly_returns.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Drawdown plot
        plt.figure(figsize=(12, 8))
        cumulative_returns = (1 + returns_df['returns']).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns / running_max) - 1
        
        plt.fill_between(drawdowns.index, drawdowns * 100, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        plt.plot(drawdowns.index, drawdowns * 100, color='darkred', linewidth=1)
        plt.title('Strategy Drawdown (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'drawdown.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to {plots_dir}")
    
    def compare_strategies(self, other_returns: Dict[str, pd.DataFrame], 
                          output_dir: str = "result") -> None:
        """
        Compare multiple strategies
        
        Args:
            other_returns (Dict[str, pd.DataFrame]): Dictionary of strategy names and returns
            output_dir (str): Output directory for comparison results
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_strategy() first.")
            return
        
        comparison_dir = os.path.join(output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Prepare comparison data
        comparison_data = {
            'Current Strategy': self.evaluation_results['returns_df']['returns']
        }
        comparison_data.update(other_returns)
        
        # Calculate metrics for all strategies
        comparison_metrics = []
        for name, returns in comparison_data.items():
            if isinstance(returns, pd.DataFrame) and 'returns' in returns.columns:
                ret_series = returns['returns']
            else:
                ret_series = returns
            
            annual_ret, sharpe, mdd = calculate_shapre_and_mdd(
                pd.DataFrame({'returns': ret_series}),
                self.risk_free_rate,
                self.trading_days
            )
            
            comparison_metrics.append({
                'Strategy': name,
                'Annual Return': annual_ret,
                'Sharpe Ratio': sharpe,
                'Max Drawdown': mdd
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_metrics)
        comparison_df.set_index('Strategy', inplace=True)
        
        # Save comparison results
        comparison_file = os.path.join(comparison_dir, "strategy_comparison.csv")
        comparison_df.to_csv(comparison_file)
        print(f"Strategy comparison saved to {comparison_file}")
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        for name, returns in comparison_data.items():
            if isinstance(returns, pd.DataFrame) and 'returns' in returns.columns:
                ret_series = returns['returns']
            else:
                ret_series = returns
            
            cumulative = (1 + ret_series).cumprod()
            plt.plot(cumulative.index, cumulative.values, label=name, linewidth=2)
        
        plt.title('Strategy Comparison - Cumulative Returns', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'strategy_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Strategy comparison plots saved to {comparison_dir}")


def create_evaluator(risk_free_rate: float = 0.05, 
                    trading_days: int = 252) -> StrategyEvaluator:
    """
    Factory function to create a StrategyEvaluator instance
    
    Args:
        risk_free_rate (float): Annual risk-free rate
        trading_days (int): Number of trading days per year
        
    Returns:
        StrategyEvaluator: Configured evaluator instance
    """
    return StrategyEvaluator(risk_free_rate=risk_free_rate, trading_days=trading_days)


if __name__ == "__main__":
    # Example usage
    evaluator = create_evaluator()
    
    # Example returns data (replace with actual strategy returns)
    dates = pd.date_range('2021-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))  # Example returns
    
    returns_df = pd.DataFrame({
        'returns': returns
    }, index=dates)
    
    # Evaluate strategy
    results = evaluator.evaluate_strategy(returns_df, plotting=True)
    
    # Generate reports
    evaluator.generate_performance_report()
    evaluator.plot_performance_analysis()
    
    print("Evaluation completed!")
