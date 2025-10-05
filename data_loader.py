"""
Data loader module for loading and preprocessing financial data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database.data_service import get_stock_data, get_vn30
from filter.financial import StockFilter
import os

class DataLoader:
    """
    Data loader class for handling financial data loading and preprocessing
    """
    
    def __init__(self, start_date: str, end_date: str, 
                 estimation_window: int = 60, 
                 correlation_threshold: float = 0.6):
        """
        Initialize the data loader
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            estimation_window (int): Estimation window for data
            correlation_threshold (float): Correlation threshold for stock selection
        """
        self.start_date = start_date
        self.end_date = end_date
        self.estimation_window = estimation_window
        self.correlation_threshold = correlation_threshold
        
        # Initialize stock filter
        self.stock_filter = StockFilter(correlation_threshold=correlation_threshold)
        
        # Data storage
        self.stock_data = None
        self.vn30_data = None
        self.selected_stocks = None
    
    def load_stock_data(self, symbols: list = None) -> pd.DataFrame:
        """
        Load stock price data for specified symbols
        
        Args:
            symbols (list): List of stock symbols to load
            
        Returns:
            pd.DataFrame: Stock price data
        """
        if symbols is None:
            # Use default symbols
            symbols = ['VN30F1M', 'VIC', 'VCB', 'VHM', 'VNM', 'BID']
        
        print(f"Loading stock data from {self.start_date} to {self.end_date}")
        self.stock_data = get_stock_data(symbols, self.start_date, self.end_date)
        
        print(f"Loaded data shape: {self.stock_data.shape}")
        print(f"Date range: {self.stock_data.index.min()} to {self.stock_data.index.max()}")
        
        return self.stock_data
    
    def load_vn30_data(self) -> pd.DataFrame:
        """
        Load VN30 data
        
        Returns:
            pd.DataFrame: VN30 data
        """
        print(f"Loading VN30 data from {self.start_date} to {self.end_date}")
        self.vn30_data = get_vn30(self.start_date, self.end_date)
        
        if self.vn30_data is not None:
            print(f"Loaded VN30 data shape: {self.vn30_data.shape}")
            print(f"Date range: {self.vn30_data.index.min()} to {self.vn30_data.index.max()}")
        
        return self.vn30_data
    
    def select_arbitrage_stocks(self) -> list:
        """
        Select stocks for arbitrage strategy
        
        Returns:
            list: Selected stock symbols
        """
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call load_stock_data() first.")
        
        # Extract futures data
        futures_data = self.stock_data['VN30F1M'] if 'VN30F1M' in self.stock_data.columns else None
        
        # Select stocks using the filter
        self.selected_stocks = self.stock_filter.select_arbitrage_stocks(
            self.stock_data.drop('VN30F1M', axis=1, errors='ignore'),
            futures_data
        )
        
        print(f"Selected stocks for arbitrage: {self.selected_stocks}")
        return self.selected_stocks
    
    def get_processed_data(self) -> tuple:
        """
        Get processed data for strategy
        
        Returns:
            tuple: (stock_data, futures_data, selected_stocks)
        """
        if self.stock_data is None:
            self.load_stock_data()
        
        if self.selected_stocks is None:
            self.select_arbitrage_stocks()
        
        # Extract futures data
        futures_data = self.stock_data['VN30F1M'] if 'VN30F1M' in self.stock_data.columns else None
        
        # Filter stock data to selected stocks
        stock_columns = [col for col in self.selected_stocks if col in self.stock_data.columns]
        filtered_stock_data = self.stock_data[stock_columns]
        
        return filtered_stock_data, futures_data, self.selected_stocks
    
    def save_data(self, output_dir: str = "result") -> None:
        """
        Save loaded data to files
        
        Args:
            output_dir (str): Output directory for saving data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.stock_data is not None:
            stock_file = os.path.join(output_dir, "stock_data.csv")
            self.stock_data.to_csv(stock_file)
            print(f"Stock data saved to {stock_file}")
        
        if self.vn30_data is not None:
            vn30_file = os.path.join(output_dir, "vn30_data.csv")
            self.vn30_data.to_csv(vn30_file)
            print(f"VN30 data saved to {vn30_file}")
        
        if self.selected_stocks is not None:
            stocks_file = os.path.join(output_dir, "selected_stocks.txt")
            with open(stocks_file, 'w') as f:
                f.write('\n'.join(self.selected_stocks))
            print(f"Selected stocks saved to {stocks_file}")
    
    def load_from_files(self, data_dir: str = "result") -> bool:
        """
        Load data from previously saved files
        
        Args:
            data_dir (str): Directory containing saved data files
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            stock_file = os.path.join(data_dir, "stock_data.csv")
            vn30_file = os.path.join(data_dir, "vn30_data.csv")
            stocks_file = os.path.join(data_dir, "selected_stocks.txt")
            
            if os.path.exists(stock_file):
                self.stock_data = pd.read_csv(stock_file, index_col=0, parse_dates=True)
                print(f"Loaded stock data from {stock_file}")
            
            if os.path.exists(vn30_file):
                self.vn30_data = pd.read_csv(vn30_file, index_col=0, parse_dates=True)
                print(f"Loaded VN30 data from {vn30_file}")
            
            if os.path.exists(stocks_file):
                with open(stocks_file, 'r') as f:
                    self.selected_stocks = f.read().strip().split('\n')
                print(f"Loaded selected stocks from {stocks_file}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data from files: {e}")
            return False


def create_data_loader(start_date: str, end_date: str, 
                      estimation_window: int = 60,
                      correlation_threshold: float = 0.6) -> DataLoader:
    """
    Factory function to create a DataLoader instance
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        estimation_window (int): Estimation window for data
        correlation_threshold (float): Correlation threshold for stock selection
        
    Returns:
        DataLoader: Configured data loader instance
    """
    return DataLoader(
        start_date=start_date,
        end_date=end_date,
        estimation_window=estimation_window,
        correlation_threshold=correlation_threshold
    )


if __name__ == "__main__":
    # Example usage
    loader = create_data_loader(
        start_date="2021-06-01",
        end_date="2024-12-31",
        estimation_window=60,
        correlation_threshold=0.6
    )
    
    # Load data
    stock_data = loader.load_stock_data()
    vn30_data = loader.load_vn30_data()
    selected_stocks = loader.select_arbitrage_stocks()
    
    # Get processed data
    processed_stocks, futures_data, stocks = loader.get_processed_data()
    
    print(f"Processed data shape: {processed_stocks.shape}")
    print(f"Futures data shape: {futures_data.shape if futures_data is not None else 'None'}")
    print(f"Selected stocks: {stocks}")
