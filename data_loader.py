"""
Load VN30F1 futures and stock_05 data
"""

import os
import json
from datetime import datetime
import pandas as pd
from database.data_service import DataService

# Load backtesting parameters from JSON file
with open("parameter/backtesting_parameter.json", 'r', encoding="utf-8") as f:
    BACKTESTING_CONFIG = json.load(f)

def init_folder(path: str):
    """Create folder if not exists"""
    os.makedirs(path, exist_ok=True)

def load_vn30f1_data(from_date, to_date, validation=False):
    """Load VN30F1 futures data"""
    data_service = DataService()
    cursor = data_service.connection.cursor()
    
    print(f"Loading VN30F1 futures data...")
    print(f"Date range: {from_date} to {to_date}")
    
    # Use VN30F1M contract
    cursor.execute("""
        SELECT c.datetime, c.price
        FROM quote.close c
        JOIN quote.futurecontractcode fc 
            ON c.datetime = fc.datetime 
            AND fc.tickersymbol = c.tickersymbol
        WHERE fc.futurecode = 'VN30F1M'
            AND c.datetime BETWEEN %s AND %s
        ORDER BY c.datetime
    """, (str(from_date), str(to_date)))
    
    results = cursor.fetchall()
    cursor.close()
    
    if not results:
        print("❌ No VN30F1 futures data found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results, columns=['datetime', 'close'])
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    
    # Get daily close price
    daily_close = df.groupby('date')['close'].last().reset_index()
    daily_close['date'] = pd.to_datetime(daily_close['date'])
    daily_close = daily_close.set_index('date')
    daily_close.columns = ['Stock']
    
    # Save to file
    output_path = f"data/os/vn30f1.csv" if validation else f"data/is/vn30f1.csv"
    daily_close.to_csv(output_path)
    
    print(f"✅ Saved VN30F1 data to {output_path}")
    print(f"   Data shape: {daily_close.shape}")
    print(f"   Date range: {daily_close.index.min()} to {daily_close.index.max()}")

def load_stock_05_data(from_date, to_date, validation=False):
    """Load stock data for 5 stocks: VIC, VCB, VHM, VNM, BID"""
    data_service = DataService()
    
    print(f"Loading stock_05 data...")
    print(f"Date range: {from_date} to {to_date}")
    
    # Get daily data
    daily_data = data_service.get_daily_data(str(from_date), str(to_date))
    
    if daily_data.empty:
        print("❌ No stock data found")
        return
    
    # Filter for target stocks
    target_stocks = ['VIC', 'VCB', 'VHM', 'VNM', 'BID']
    filtered_data = daily_data[daily_data['tickersymbol'].isin(target_stocks)].copy()
    
    if filtered_data.empty:
        print("❌ No data found for target stocks")
        return
    
    # Process data
    filtered_data['date'] = pd.to_datetime(filtered_data['date']).dt.date
    pivot_data = filtered_data.pivot_table(
        index='date', 
        columns='tickersymbol', 
        values='close', 
        aggfunc='last'
    )
    
    # Ensure all stocks are present
    for stock in target_stocks:
        if stock not in pivot_data.columns:
            pivot_data[stock] = None
    
    pivot_data = pivot_data[target_stocks]
    pivot_data = pivot_data.reset_index()
    pivot_data['date'] = pd.to_datetime(pivot_data['date'])
    pivot_data = pivot_data.set_index('date')
    
    # Save to file
    output_path = f"data/os/stock_05.csv" if validation else f"data/is/stock_05.csv"
    pivot_data.to_csv(output_path)
    
    print(f"✅ Saved stock_05 data to {output_path}")
    print(f"   Data shape: {pivot_data.shape}")
    print(f"   Date range: {pivot_data.index.min()} to {pivot_data.index.max()}")

def sync_data_dates():
    """Sync data dates between VN30F1 and stock_05"""
    
    for period in ['is', 'os']:
        
        # File paths
        vn30f1_path = f"data/{period}/vn30f1.csv"
        stock_path = f"data/{period}/stock_05.csv"
        
        # Check if files exist
        if not os.path.exists(vn30f1_path) or not os.path.exists(stock_path):
            print(f"❌ Files not found for {period}")
            continue
            
        # Load data
        vn30f1_data = pd.read_csv(vn30f1_path, index_col=0, parse_dates=True)
        stock_data = pd.read_csv(stock_path, index_col=0, parse_dates=True)
        
        # Find common dates
        common_dates = vn30f1_data.index.intersection(stock_data.index)
        
        if len(common_dates) == 0:
            print(f"   ❌ No common dates found!")
            continue
            
        # Filter both datasets to common dates
        vn30f1_synced = vn30f1_data.loc[common_dates].sort_index()
        stock_synced = stock_data.loc[common_dates].sort_index()
    
        # Save synced data
        vn30f1_synced.to_csv(vn30f1_path)
        stock_synced.to_csv(stock_path)
        

if __name__ == "__main__":
    # Create required directories
    required_directories = [
        "data",
        "data/is",
        "data/os",
        "result/optimization",
        "result/backtest",
    ]
    for dr in required_directories:
        init_folder(dr)
    
    # Load date ranges from config
    is_from_date_str = BACKTESTING_CONFIG["is_from_date_str"]
    is_to_date_str = BACKTESTING_CONFIG["is_end_date_str"]
    os_from_date_str = BACKTESTING_CONFIG["os_from_date_str"]
    os_to_date_str = BACKTESTING_CONFIG["os_to_date_str"]
    
    is_from_date = datetime.strptime(is_from_date_str, "%Y-%m-%d").date()
    is_to_date = datetime.strptime(is_to_date_str, "%Y-%m-%d").date()
    os_from_date = datetime.strptime(os_from_date_str, "%Y-%m-%d").date()
    os_to_date = datetime.strptime(os_to_date_str, "%Y-%m-%d").date()
    
    print("🚀 Loading VN30F1 and Stock_05 Data")
    print("=" * 60)
    
    # Load in-sample data
    print("\n📊 Loading in-sample data...")
    load_vn30f1_data(is_from_date, is_to_date)
    load_stock_05_data(is_from_date, is_to_date)
    
    # Load out-of-sample data
    print("\n📊 Loading out-of-sample data...")
    load_vn30f1_data(os_from_date, os_to_date, validation=True)
    load_stock_05_data(os_from_date, os_to_date, validation=True)
    
    # Sync data dates
    sync_data_dates()
    
    print("\n✅ Data loading completed!")
    print("=" * 60)
