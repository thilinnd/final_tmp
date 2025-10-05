"""
This module is used for calculating metric
"""

from typing import List
from decimal import Decimal
import numpy as np
import pandas as pd


def get_returns(
    monthly_df: pd.DataFrame,
    index_df: pd.DataFrame,
):
    """
    Get multiple period returns

    Args:
        monthly_df (pd.DataFrame): _description_
        index_df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    monthly_df["monthly_return"] = monthly_df["asset"].copy().pct_change()
    index_df["index_monthly_return"] = (
        index_df["ac_return"]
        .rolling(2)
        .apply(lambda x: ((x.iloc[1] + 1) / (x.iloc[0] + 1)) - 1)
    ).copy()
    monthly_df = monthly_df.astype({"monthly_return": float})
    merged_monthly_index = pd.merge(monthly_df, index_df, on=["date"])
    merged_monthly_index["exess_monthly_return"] = (
        merged_monthly_index["monthly_return"].copy()
        - merged_monthly_index["index_monthly_return"].copy()
    )

    annual_return = (
        np.prod(1 + merged_monthly_index["monthly_return"])
        ** (12 / len(merged_monthly_index["monthly_return"]))
        - 1
    )

    return {
        "annual_return": annual_return,
        "monthly_return": merged_monthly_index['monthly_return'].mean(),
        "excess_monthly_return": merged_monthly_index['exess_monthly_return'].mean(),
    }


class Metric:
    """
    Metric: sharpe, sortino, information ratios, MDD
    """

    def __init__(self, period_returns: List[Decimal], benchmark_returns: List[Decimal]):
        """
        Args:
            period_returns (List[Decimal]): _description_
            benchmark_returns (List[Decimal]): _description_
        """
        self.period_returns = period_returns
        self.benchmark_returns = benchmark_returns

    def hpr(self) -> Decimal:
        return (np.cumprod(1 + np.array(self.period_returns)) - 1)[-1]

    def excess_hpr(self) -> Decimal:
        return (
            np.cumprod(1 + np.array(self.period_returns))
            - np.cumprod(1 + np.array(self.benchmark_returns))
        )[-1]

    def sharpe_ratio(self, risk_free_return: Decimal) -> Decimal:
        """
        Calculate sharpe ratio

        Args:
            risk_free_return (Decimal): _description_

        Raises:
            ValueError: None or empty period returns

        Returns:
            Decimal
        """
        if not self.period_returns:
            raise ValueError('Annual returns should not be None or empty')

        # Calculate excess returns
        excess_returns = [
            period_return - risk_free_return for period_return in self.period_returns
        ]

        return np.mean(excess_returns) / np.std(self.period_returns, ddof=1)

    def sortino_ratio(self, risk_free_return: Decimal) -> Decimal:
        """
        Calculate sortino ratio

        Args:
            risk_free_return (Decimal): _description_

        Raises:
            ValueError: None or empty period returns

        Returns:
            Decimal: _description_
        """
        if not self.period_returns:
            raise ValueError('Annual returns should not be None or empty')

        downside_returns = [
            min(0, period_return - risk_free_return)
            for period_return in self.period_returns
        ]
        downside_risk = np.sqrt(np.mean([d_r**2 for d_r in downside_returns]))

        return (np.mean(self.period_returns) - risk_free_return) / downside_risk

    def maximum_drawdown(self) -> Decimal:
        """
        Calculate maximum drawdown

        Raises:
            ValueError: None or empty period returns
            ValueError: Invalid input

        Returns:
            Decimal: _description_
        """
        dds = []
        if not self.period_returns:
            raise ValueError('Invalid Input')

        if any(period_return <= -1 for period_return in self.period_returns):
            raise ValueError('Invalid Input')

        peak = 1
        cur_perf = 1
        mdd = 0
        for period_return in self.period_returns:
            cur_perf *= 1 + period_return
            peak = max(peak, cur_perf)

            dd = cur_perf / peak - 1
            dds.append(dd)
            mdd = min(dd, mdd)

        return mdd, dds

    def longest_drawdown(self) -> int:
        """
        Calculate longest drawdown

        Raises:
            ValueError: None of empty period returns
            ValueError: Invalid

        Returns:
            int: _description_
        """
        if not self.period_returns:
            raise ValueError('Invalid Input')

        if any(period_return <= -1 for period_return in self.period_returns):
            raise ValueError('Invalid Input')

        cur_period = 0
        max_period = 0
        peak = 1
        cur_perf = 1
        for period_return in self.period_returns:
            cur_perf *= 1 + period_return

            if cur_perf > peak:
                cur_period = 0
                peak = cur_perf
                continue

            cur_period += 1
            max_period = max(max_period, cur_period)

        return max_period

    def information_ratio(self) -> Decimal:
        """
        Calculate informaton ratio

        Raises:
            ValueError: None or empty period return or benchmark return
            ValueError: Not equal length
            ValueError: Invalid input
            ValueError: Invalid length

        Returns:
            Decimal: _description_
        """
        if not self.period_returns or not self.benchmark_returns:
            raise ValueError("Invalid Input")

        if len(self.period_returns) != len(self.benchmark_returns):
            raise ValueError(
                f"Not equal length {len(self.period_returns)} - {len(self.benchmark_returns)}"
            )

        if any(period_return <= -1 for period_return in self.period_returns) or any(
            benchmark_return <= -1 for benchmark_return in self.benchmark_returns
        ):
            raise ValueError("Invalid Input")

        if len(self.period_returns) == 1 or len(self.benchmark_returns) == 1:
            raise ValueError("Invalid length")

        mean_period_returns = np.array(self.period_returns).mean()
        mean_benchmark_returns = np.array(self.benchmark_returns).mean()

        if mean_period_returns == mean_benchmark_returns:
            return 0

        excess_returns = np.array(self.period_returns) - np.array(
            self.benchmark_returns
        )

        return (mean_period_returns - mean_benchmark_returns) / excess_returns.std()


# Additional functions needed by evaluation.py
import os
import matplotlib.pyplot as plt
from datetime import datetime
from database.data_service import get_etf_price
from tabulate import tabulate


def calculate_cumulative_returns(returns_series: pd.Series) -> pd.Series:
    """Calculate cumulative returns from a series of returns."""
    return (1 + returns_series).cumprod()


def calculate_hpr(cumulative_returns: pd.Series) -> float:
    """Calculate the Holding Period Return (HPR)."""
    return cumulative_returns.iloc[-1] - 1


def calculate_annualized_return(hpr: float, time_length: float) -> float:
    """Calculate the annualized return."""
    return (1 + hpr) ** (1 / time_length) - 1


def calculate_excess_hpr(strategy_hpr: float, benchmark_hpr: float) -> float:
    """Calculate the excess Holding Period Return (HPR)."""
    return strategy_hpr - benchmark_hpr


def calculate_annual_excess_return(strategy_annual_return: float, benchmark_annual_return: float) -> float:
    """Calculate the annualized excess return."""
    return strategy_annual_return - benchmark_annual_return


def calculate_volatility(returns_series: pd.Series, trading_days: int) -> float:
    """Calculate the annualized volatility."""
    return returns_series.std() * np.sqrt(trading_days)


def calculate_drawdowns(cumulative_returns: pd.Series) -> pd.Series:
    """Calculate the drawdown series."""
    running_max = np.maximum.accumulate(cumulative_returns.dropna())
    running_max[running_max < 1] = 1
    drawdowns = (cumulative_returns / running_max) - 1
    return drawdowns


def calculate_max_drawdown(drawdowns: pd.Series) -> float:
    """Calculate the maximum drawdown."""
    return -drawdowns.min()


def calculate_longest_drawdown(cumulative_returns: pd.Series) -> int:
    """Calculate the longest drawdown period in days."""
    drawdowns = calculate_drawdowns(cumulative_returns)
    in_drawdown = drawdowns < 0
    drawdown_periods = []
    start = None
    for i, is_dd in enumerate(in_drawdown):
        if is_dd and start is None:
            start = i
        elif not is_dd and start is not None:
            drawdown_periods.append(i - start)
            start = None
    if start is not None:
        drawdown_periods.append(len(in_drawdown) - start)
    return max(drawdown_periods) if drawdown_periods else 0


def calculate_sharpe_ratio(annual_return: float, volatility: float, risk_free_rate: float) -> float:
    """Calculate the Sharpe Ratio."""
    return (annual_return - risk_free_rate) / volatility if volatility != 0 else np.nan


def calculate_downside_deviation(returns_series: pd.Series, trading_days: int) -> float:
    """Calculate the annualized downside deviation."""
    downward = returns_series[returns_series < 0]
    return downward.std() * np.sqrt(trading_days) if not downward.empty else 0


def calculate_sortino_ratio(annual_return: float, downside_deviation: float) -> float:
    """Calculate the Sortino Ratio."""
    return annual_return / downside_deviation if downside_deviation != 0 else np.nan


def calculate_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate the beta of the asset relative to the benchmark."""
    cov = np.cov(asset_returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns)
    return cov / var if var != 0 else np.nan


def calculate_alpha(annual_return: float, beta: float, benchmark_annual_return: float) -> float:
    """Calculate the alpha of the asset."""
    return annual_return - beta * benchmark_annual_return if not np.isnan(beta) else np.nan


def calculate_tracking_error(asset_returns: pd.Series, benchmark_returns: pd.Series, trading_days: int) -> float:
    """Calculate the annualized tracking error."""
    excess_returns = asset_returns - benchmark_returns
    return excess_returns.std() * np.sqrt(trading_days)


def calculate_information_ratio(annual_excess_return: float, tracking_error: float) -> float:
    """Calculate the Information Ratio."""
    return annual_excess_return / tracking_error if tracking_error != 0 else np.nan


def calculate_turnover(total_fee: float) -> float:
    """Calculate the turnover ratio based on total fees."""
    return total_fee / (.23/100)/252


def calculate_metrics(
    returns_df: pd.DataFrame,
    total_fee_ratio,
    risk_free_rate: float = 0.05,
    trading_day: int = 252,
    freq: str = "D",
    plotting: bool = False,
    use_benchmark: bool = True,
    use_existing_data: bool = True
) -> pd.DataFrame:
    """Calculate performance metrics, plot cumulative returns, and drawdown for portfolio returns vs a benchmark."""
    # Validate inputs
    if returns_df.empty:
        raise ValueError("returns_df is empty.")
    if 'returns' not in returns_df.columns:
        raise ValueError("returns_df must contain a 'returns' column.")

    # Ensure datetime index
    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index)

    # Initialize benchmark variables
    benchmark_returns = None
    benchmark_df = None

    if use_benchmark:
        # Get the directory structure
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(module_dir)  # Go up to project root
        data_folder = os.path.join(project_root, "result")
        os.makedirs(data_folder, exist_ok=True)
        csv_file = os.path.join(data_folder, "index_price.csv")

        if use_existing_data and os.path.exists(csv_file):
            # Load existing benchmark data
            benchmark_df = pd.read_csv(csv_file, parse_dates=True)
            if 'price' not in benchmark_df.columns:
                raise ValueError("index_price.csv must contain a 'price' column.")
            benchmark_df.set_index('datetime', inplace=True)
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
        else:
            # Fetch benchmark data (assuming get_etf_price is available)
            start = returns_df.index[0].strftime('%Y-%m-%d')
            end = returns_df.index[-1].strftime('%Y-%m-%d')
            benchmark_df = get_etf_price('VN30', start, end)
            benchmark_df.set_index('datetime', inplace=True)
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
            if 'price' not in benchmark_df.columns:
                raise ValueError("Benchmark DataFrame must contain a 'price' column.")
            # Optionally save to file for future use
            benchmark_df.to_csv(csv_file)

        # Align and calculate benchmark returns
        benchmark_df = benchmark_df.reindex(returns_df.index, method='ffill')
        benchmark_returns = benchmark_df['price'].pct_change().fillna(0)

    # Calculate time length
    time_length = (returns_df.index[-1] - returns_df.index[0]).days / 365.25

    # Cumulative returns
    strategy_cum_rets = calculate_cumulative_returns(returns_df['returns'])
    benchmark_cum_rets = calculate_cumulative_returns(benchmark_returns) if use_benchmark else None

    # HPR
    strategy_hpr = calculate_hpr(strategy_cum_rets)
    benchmark_hpr = calculate_hpr(benchmark_cum_rets) if use_benchmark else None
    excess_hpr = calculate_excess_hpr(strategy_hpr, benchmark_hpr) if use_benchmark else None

    # Annualized returns
    strategy_annual_return = calculate_annualized_return(strategy_hpr, time_length)
    benchmark_annual_return = calculate_annualized_return(benchmark_hpr, time_length) if use_benchmark else None
    annual_excess_return = calculate_annual_excess_return(strategy_annual_return, benchmark_annual_return) if use_benchmark else None

    # Volatility
    strategy_volatility = calculate_volatility(returns_df['returns'], trading_day)
    benchmark_volatility = calculate_volatility(benchmark_returns, trading_day) if use_benchmark else None

    # Drawdowns and related metrics
    strategy_drawdowns = calculate_drawdowns(strategy_cum_rets)
    strategy_max_drawdown = calculate_max_drawdown(strategy_drawdowns)
    strategy_longest_drawdown = calculate_longest_drawdown(strategy_cum_rets)
    if use_benchmark:
        benchmark_drawdowns = calculate_drawdowns(benchmark_cum_rets)
        benchmark_max_drawdown = calculate_max_drawdown(benchmark_drawdowns)
        benchmark_longest_drawdown = calculate_longest_drawdown(benchmark_cum_rets)
    else:
        benchmark_drawdowns = benchmark_max_drawdown = benchmark_longest_drawdown = None

    # Risk-adjusted metrics
    strategy_sharpe = calculate_sharpe_ratio(strategy_annual_return, strategy_volatility, risk_free_rate)
    strategy_downside_dev = calculate_downside_deviation(returns_df['returns'], trading_day)
    strategy_sortino = calculate_sortino_ratio(strategy_annual_return, strategy_downside_dev)
    if use_benchmark:
        benchmark_sharpe = calculate_sharpe_ratio(benchmark_annual_return, benchmark_volatility, risk_free_rate)
        benchmark_downside_dev = calculate_downside_deviation(benchmark_returns, trading_day)
        benchmark_sortino = calculate_sortino_ratio(benchmark_annual_return, benchmark_downside_dev)
    else:
        benchmark_sharpe = benchmark_downside_dev = benchmark_sortino = None

    # Benchmark comparison metrics
    beta = calculate_beta(returns_df['returns'], benchmark_returns) if use_benchmark else None
    alpha = calculate_alpha(strategy_annual_return, beta, benchmark_annual_return) if use_benchmark else None
    tracking_error = calculate_tracking_error(returns_df['returns'], benchmark_returns, trading_day) if use_benchmark else None
    information_ratio = calculate_information_ratio(annual_excess_return, tracking_error) if use_benchmark else None

    # Turnover ratio
    turnover = calculate_turnover(total_fee_ratio)

    # Compile metrics
    metrics_data = {
        'HPR (%)': [f"{strategy_hpr * 100:.2f}%", f"{benchmark_hpr * 100:.2f}%" if use_benchmark else "-"],
        'Excess HPR (%)': [f"{excess_hpr * 100:.2f}%" if use_benchmark else "-", "-"],
        'Annual Return (%)': [f"{strategy_annual_return * 100:.2f}%", f"{benchmark_annual_return * 100:.2f}%" if use_benchmark else "-"],
        'Annual Excess Return (%)': [f"{annual_excess_return * 100:.2f}%" if use_benchmark else "-", "-"],
        'Volatility (%)': [f"{strategy_volatility * 100:.2f}%", f"{benchmark_volatility * 100:.2f}%" if use_benchmark else "-"],
        'Maximum Drawdown (%)': [f"{strategy_max_drawdown * 100:.2f}%", f"{benchmark_max_drawdown * 100:.2f}%" if use_benchmark else "-"],
        'Longest Drawdown (days)': [f"{strategy_longest_drawdown:.0f}", f"{benchmark_longest_drawdown:.0f}" if use_benchmark else "-"],
        'Sharpe Ratio': [f"{strategy_sharpe:.2f}", f"{benchmark_sharpe:.2f}" if use_benchmark else "-"],
        'Sortino Ratio': [f"{strategy_sortino:.2f}", f"{benchmark_sortino:.2f}" if use_benchmark else "-"],
        'Information Ratio': [f"{information_ratio:.2f}" if use_benchmark else "-", "-"],
        'Beta': [f"{beta:.2f}" if use_benchmark else "-", "-"],
        'Alpha (%)': [f"{alpha * 100:.2f}%" if use_benchmark else "-", "-"],
        'Turnover Ratio (%)': [f"{turnover * 100:.2f}%", "-"],
    }
    metrics_df = pd.DataFrame(metrics_data, index=['Strategy', 'VN30'] if use_benchmark else ['Strategy'])

    return metrics_df


def calculate_shapre_and_mdd(returns_df: pd.DataFrame, risk_free_rate: float = 0.05, trading_day: int = 252, freq: str = "D") -> pd.DataFrame:
    """Calculate performance metrics, plot cumulative returns, and drawdown for portfolio returns vs a benchmark."""
    # Validate inputs
    if returns_df.empty:
        raise ValueError("returns_df is empty.")
    if 'returns' not in returns_df.columns:
        raise ValueError("returns_df must contain a 'returns' column.")

    # Ensure datetime index
    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index)
    
    # Calculate time length
    time_length = (returns_df.index[-1] - returns_df.index[0]).days / 365.25
    
    # Cumulative returns
    strategy_cum_rets = calculate_cumulative_returns(returns_df['returns'])
    
    # HPR
    strategy_hpr = calculate_hpr(strategy_cum_rets)
    
    # Annualized returns
    strategy_annual_return = calculate_annualized_return(strategy_hpr, time_length)
    
    # Volatility
    strategy_volatility = calculate_volatility(returns_df['returns'], trading_day)
    
    # Drawdowns and related metrics
    strategy_drawdowns = calculate_drawdowns(strategy_cum_rets)
    strategy_max_drawdown = calculate_max_drawdown(strategy_drawdowns)
    
    # Risk-adjusted metrics
    strategy_sharpe = calculate_sharpe_ratio(strategy_annual_return, strategy_volatility, risk_free_rate)
    
    return strategy_annual_return, strategy_sharpe, strategy_max_drawdown


def calculate_monthly_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly returns from a time series of daily returns."""
    if returns_df.empty:
        raise ValueError("returns_df is empty.")
    if 'returns' not in returns_df.columns:
        raise ValueError("returns_df must contain a 'returns' column.")

    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df['Year'] = returns_df.index.year
    returns_df['Month'] = returns_df.index.month

    monthly_returns = []
    for (year, month), group in returns_df.groupby(['Year', 'Month']):
        cum_return = (1 + group['returns']).prod() - 1
        monthly_returns.append({'Year': year, 'Month': month, 'Monthly Return': cum_return})

    return pd.DataFrame(monthly_returns).sort_values(['Year', 'Month'])


def calculate_yearly_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly returns from a time series of daily returns."""
    if returns_df.empty:
        raise ValueError("returns_df is empty.")
    if 'returns' not in returns_df.columns:
        raise ValueError("returns_df must contain a 'returns' column.")

    returns_df = returns_df.copy()
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df['Year'] = returns_df.index.year

    yearly_returns = []
    for year, group in returns_df.groupby('Year'):
        cum_return = (1 + group['returns']).prod() - 1
        yearly_returns.append({'Year': year, 'Yearly Return': cum_return})

    return pd.DataFrame(yearly_returns)


def pivot_monthly_returns_to_table(monthly_returns_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot monthly returns into a table with years as columns and months as rows."""
    if monthly_returns_df.empty:
        raise ValueError("monthly_returns_df is empty.")
    if not all(col in monthly_returns_df.columns for col in ['Year', 'Month', 'Monthly Return']):
        raise ValueError("monthly_returns_df must contain 'Year', 'Month', and 'Monthly Return' columns.")

    # Map month numbers to month names
    month_map = {
        1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
        7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
    }
    monthly_returns_df = monthly_returns_df.copy()
    monthly_returns_df['Month'] = monthly_returns_df['Month'].map(month_map)

    # Add a yearly row for totals
    yearly_returns = monthly_returns_df.groupby('Year')['Monthly Return'].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    yearly_returns['Month'] = 'YEARLY'
    yearly_returns = yearly_returns[['Year', 'Month', 'Monthly Return']]

    # Concatenate the yearly returns with the monthly returns
    monthly_returns_df = pd.concat([yearly_returns, monthly_returns_df], ignore_index=True)

    # Pivot the table
    pivoted_df = monthly_returns_df.pivot(index='Month', columns='Year', values='Monthly Return')

    # Ensure all months and the 'YEARLY' row are present in the correct order
    desired_index = ['YEARLY', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                     'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    pivoted_df = pivoted_df.reindex(desired_index)

    # Format the values as percentages with 3 decimal places
    pivoted_df = pivoted_df.map(lambda x: f"{x:.2%}" if pd.notnull(x) else x)

    return pivoted_df