"""
Database data service module
"""

import pandas as pd
import psycopg2
import time
from .query import STOCK_PRICE_QUERY, ETF_PRICE_QUERY, FUTURES_PRICE_QUERY, VN30_QUERY
from config.config import db_params

# Establish global connection (consider passing as parameter in production)
CONNECTION = psycopg2.connect(**db_params)


def execute_query(query, from_date, to_date):
    """Execute a database query with date range parameters.

    Args:
        query (str): SQL query with %s placeholders for dates.
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        list: Query results as a list of tuples, or None if an error occurs.
    """
    cursor = CONNECTION.cursor()
    try:
        cursor.execute(query, (from_date, to_date))
        result = cursor.fetchall()
        cursor.close()
        CONNECTION.commit()
        return result
    except Exception as e:
        print(f"Error: {e}")
        CONNECTION.rollback()
        cursor.close()
        return None


def get_stock_price(symbol, start_date, end_date):
    """Fetch adjusted close prices for a stock from the database.

    Args:
        symbol (str): Stock ticker symbol (e.g., 'ACB').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'datetime', 'tickersymbol', 'price' columns,
                      or None if query fails.
    """
    cursor = CONNECTION.cursor()
    try:
        cursor.execute(STOCK_PRICE_QUERY, (symbol, start_date, end_date))
        result = cursor.fetchall()
        cursor.close()
        CONNECTION.commit()
        columns = ["datetime", "tickersymbol", "price"]
        matched = pd.DataFrame(result, columns=columns)
        matched = matched.astype({"price": float})
        return matched
    except Exception as e:
        print(f"Error: {e}")
        CONNECTION.rollback()
        cursor.close()
        return None


def get_etf_price(symbol, start_date, end_date):
    """Fetch close prices for an ETF from the database.

    Args:
        symbol (str): ETF ticker symbol (e.g., 'FUEVFVND').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'datetime', 'tickersymbol', 'price' columns,
                      or None if query fails.
    """
    cursor = CONNECTION.cursor()
    try:
        cursor.execute(ETF_PRICE_QUERY, (symbol, start_date, end_date))
        result = cursor.fetchall()
        cursor.close()
        CONNECTION.commit()
        columns = ["datetime", "tickersymbol", "price"]
        matched = pd.DataFrame(result, columns=columns)
        matched = matched.astype({"price": float})
        return matched
    except Exception as e:
        print(f"Error: {e}")
        CONNECTION.rollback()
        cursor.close()
        return None


def get_futures_price(start_date, end_date):
    """Fetch futures price data for VN30F1M from the database.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'datetime', 'tickersymbol', 'price' columns,
                      or empty DataFrame if query fails.
    """
    cursor = CONNECTION.cursor()
    try:
        cursor.execute(FUTURES_PRICE_QUERY, (start_date, end_date))
        result = cursor.fetchall()
        cursor.close()
        CONNECTION.commit()
        columns = ["datetime", "tickersymbol", "price"]
        matched = pd.DataFrame(result, columns=columns)
        matched = matched.astype({"price": float})
        return matched
    except Exception as e:
        print(f"Error executing query: {e}")
        CONNECTION.rollback()
        cursor.close()
        return pd.DataFrame(columns=["datetime", "tickersymbol", "price"])


def get_stock_data(symbols, start_date, end_date):
    """Fetch stock, ETF, and futures price data for multiple symbols.

    Args:
        symbols (list): List of ticker symbols (e.g., ['VN30F1M', 'ACB']).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'Date' index and symbol columns containing prices.
    """
    
    stock_data = pd.DataFrame()
    for symbol in symbols:
        if symbol == 'VN30F1M':
            close_price = get_futures_price(start_date, end_date)
        elif symbol in ['FUEVFVND', 'FUESSVFL', 'E1VFVN30', 'FUEVN100']:
            close_price = get_etf_price(symbol, start_date, end_date)
        else:
            close_price = get_stock_price(symbol, start_date, end_date)
        
        if close_price is not None:
            close_price = close_price[['datetime', 'price']]
            close_price.set_index('datetime', inplace=True)
            close_price.rename(columns={'price': symbol}, inplace=True)
            stock_data = pd.concat([stock_data, close_price], axis=1).dropna()
        time.sleep(0.5)  # Rate limiting
    
    stock_data['Date'] = pd.to_datetime(stock_data.index)
    stock_data = stock_data.set_index('Date').sort_index()
    return stock_data


def get_vn30(from_date, to_date):
    """Fetch VN30 data between specified dates from the database.

    Args:
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'Date' index and 'Stock' column, or None if query fails.
    """
    cursor = CONNECTION.cursor()
    try:
        cursor.execute(VN30_QUERY, (from_date, to_date))
        result = pd.DataFrame(cursor.fetchall())
        cursor.close()
        CONNECTION.commit()
    except Exception as e:
        print(f"Error: {e}")
        CONNECTION.rollback()
        cursor.close()
        return None
    
    result = result.rename(columns={result.columns[0]: 'Date', result.columns[1]: 'Stock'})
    result_df = result.set_index('Date')
    result_df.index = pd.to_datetime(result_df.index)
    return result_df
