"""
Database data service module - Simplified for statistical arbitrage
"""

import pandas as pd
import psycopg2
from .query import DAILY_DATA_QUERY
from config.config import db_params


class DataService:
    """
    Simplified data service for statistical arbitrage
    """

    def __init__(self) -> None:
        """
        Initialize database connection
        """
        try:
            if (
                db_params["host"]
                and db_params["port"]
                and db_params["database"]
                and db_params["user"]
                and db_params["password"]
            ):
                self.connection = psycopg2.connect(**db_params)
                self.is_file = False
                print("✅ Database connection established")
            else:
                self.connection = None
                self.is_file = True
                print("⚠️ Database parameters not configured")
        except Exception as e:
            self.connection = None
            self.is_file = True
            print(f"⚠️ Database connection failed: {e}")

    def get_daily_data(
        self,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """
        Get daily stock data from database

        Args:
            from_date (str): Start date in 'YYYY-MM-DD' format
            to_date (str): End date in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: Daily stock data with columns [year, date, tickersymbol, close]
        """
        if not self.connection:
            print("❌ No database connection available")
            return pd.DataFrame()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(DAILY_DATA_QUERY, (from_date, to_date))

            queries = list(cursor)
            cursor.close()

            if not queries:
                print("❌ No daily data found")
                return pd.DataFrame()

            columns = ["year", "date", "tickersymbol", "close"]
            df = pd.DataFrame(queries, columns=columns)
            print(f"✅ Loaded {len(df)} daily records from database")
            return df
            
        except Exception as e:
            print(f"❌ Error loading daily data: {e}")
            return pd.DataFrame()

    def get_close_price(
        self,
        from_date: str,
        to_date: str,
        contract_type: str,
    ) -> pd.DataFrame:
        """
        Get close price data for futures contracts

        Args:
            from_date (str): Start date in 'YYYY-MM-DD' format
            to_date (str): End date in 'YYYY-MM-DD' format
            contract_type (str): Contract type (e.g., 'VN30F1M')

        Returns:
            pd.DataFrame: Close price data with columns [datetime, tickersymbol, close]
        """
        if not self.connection:
            print("❌ No database connection available")
            return pd.DataFrame()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT c.datetime, c.tickersymbol, c.price
                FROM quote.close c
                JOIN quote.futurecontractcode fc 
                    ON c.datetime = fc.datetime 
                    AND fc.tickersymbol = c.tickersymbol
                WHERE fc.futurecode = %s
                    AND c.datetime BETWEEN %s AND %s
                ORDER BY c.datetime
            """, (contract_type, from_date, to_date))

            results = cursor.fetchall()
            cursor.close()

            if not results:
                print(f"❌ No close price data found for {contract_type}")
                return pd.DataFrame()

            df = pd.DataFrame(results, columns=['datetime', 'tickersymbol', 'close'])
            print(f"✅ Loaded {len(df)} close price records for {contract_type}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading close price data: {e}")
            return pd.DataFrame()

    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✅ Database connection closed")

    def __del__(self):
        """Destructor to close connection"""
        self.close_connection()


# Standalone functions for backward compatibility
def get_futures_price(start_date, end_date):
    """
    Get VN30F1M futures price data (backward compatibility)
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: Futures price data
    """
    data_service = DataService()
    return data_service.get_close_price(start_date, end_date, 'VN30F1M')

def get_stock_data(symbols, start_date, end_date):
    """
    Get stock data for specified symbols (backward compatibility)
    
    Args:
        symbols (list): List of stock symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: Stock data
    """
    data_service = DataService()
    daily_data = data_service.get_daily_data(start_date, end_date)
    
    if daily_data.empty:
        return pd.DataFrame()
    
    # Filter for specified symbols
    filtered_data = daily_data[daily_data['tickersymbol'].isin(symbols)].copy()
    
    if filtered_data.empty:
        return pd.DataFrame()
    
    # Process data
    filtered_data['date'] = pd.to_datetime(filtered_data['date']).dt.date
    pivot_data = filtered_data.pivot_table(
        index='date', 
        columns='tickersymbol', 
        values='close', 
        aggfunc='last'
    )
    
    # Ensure all symbols are present
    for symbol in symbols:
        if symbol not in pivot_data.columns:
            pivot_data[symbol] = None
    
    pivot_data = pivot_data[symbols]
    pivot_data = pivot_data.reset_index()
    pivot_data['date'] = pd.to_datetime(pivot_data['date'])
    pivot_data = pivot_data.set_index('date')
    
    return pivot_data

def get_vn30(from_date, to_date):
    """
    Get VN30 index data (backward compatibility)
    
    Args:
        from_date (str): Start date in 'YYYY-MM-DD' format
        to_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: VN30 index data
    """
    data_service = DataService()
    return data_service.get_close_price(from_date, to_date, 'VN30')