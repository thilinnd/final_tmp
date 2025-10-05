"""
Phân tích vị thế với biểu đồ HPR (Holding Period Return) thay vì P&L
"""

import pandas as pd
import numpy as np
from position_manager import PositionManager, PositionType
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

def load_real_data():
    """Load dữ liệu thực tế từ backtesting"""
    try:
        # Load VN30F1M data
        vn30f1m = pd.read_csv("data/vn30f1m.csv", index_col=0, parse_dates=True)
        vn30f1m_price = vn30f1m["price"]
        
        # Load VN30 stocks data
        vn30_stocks = pd.read_csv("data/vn30_stocks.csv", index_col=0, parse_dates=True)
        
        print(f"Loaded VN30F1M data: {len(vn30f1m_price)} records")
        print(f"Loaded VN30 stocks data: {vn30_stocks.shape}")
        
        return vn30f1m_price, vn30_stocks
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def create_vn05_basket(vn30_stocks):
    """Tạo rổ VN05 từ VN30 stocks"""
    # Chọn 5 cổ phiếu chính (VIC, VCB, VHM, VNM, BID)
    selected_stocks = ['VIC', 'VCB', 'VHM', 'VNM', 'BID']
    available_stocks = [col for col in selected_stocks if col in vn30_stocks.columns]
    
    if len(available_stocks) == 0:
        print("No selected stocks available, using first 5 stocks")
        available_stocks = vn30_stocks.columns[:5].tolist()
    
    print(f"Using stocks for VN05 basket: {available_stocks}")
    
    # Tạo rổ VN05 (trung bình có trọng số)
    vn05_basket = vn30_stocks[available_stocks].mean(axis=1)
    vn05_basket.name = 'VN05'
    
    return vn05_basket, available_stocks

def calculate_spread_and_signals(vn05_prices, futures_prices, window=30):
    """Tính spread và tạo signals"""
    # Tính spread
    spread = vn05_prices - 1.2 * futures_prices
    
    # Tính z-score
    z_score = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    
    # Tạo signals
    signals = pd.Series(0, index=vn05_prices.index)
    signals[z_score < -1.5] = 1   # Long VN05, Short VN30F1M
    signals[z_score > 1.5] = -1   # Short VN05, Long VN30F1M
    
    return spread, z_score, signals

def calculate_hpr(vn05_prices, futures_prices, position_info):
    """Tính HPR (Holding Period Return) cho vị thế"""
    entry_date = position_info['date']
    position_type = position_info['type']
    entry_vn05_price = position_info['vn05_price']
    entry_futures_price = position_info['futures_price']
    hedge_ratio = position_info['hedge_ratio']
    
    # Lấy giá từ ngày entry đến hiện tại
    vn05_returns = vn05_prices.loc[entry_date:] / entry_vn05_price - 1
    futures_returns = futures_prices.loc[entry_date:] / entry_futures_price - 1
    
    # Tính HPR dựa trên loại vị thế
    if position_type == 'long_vn05_short_futures':
        # Long VN05, Short VN30F1M
        hpr = vn05_returns - hedge_ratio * futures_returns
    else:
        # Short VN05, Long VN30F1M
        hpr = -vn05_returns + hedge_ratio * futures_returns
    
    return hpr

def simulate_real_trading_with_hpr(vn05_prices, futures_prices, signals, z_score, spread):
    """Mô phỏng giao dịch với tính toán HPR"""
    print("=" * 80)
    print("MÔ PHỎNG GIAO DỊCH VỚI TÍNH TOÁN HPR")
    print("=" * 80)
    
    # Khởi tạo Position Manager
    pm = PositionManager(initial_capital=1_000_000_000, position_size=0.4)
    
    # Lưu trữ lịch sử
    trading_history = []
    position_history = []
    hpr_data = {}
    
    print(f"Bắt đầu giao dịch từ {signals.index[0].strftime('%Y-%m-%d')} đến {signals.index[-1].strftime('%Y-%m-%d')}")
    print(f"Tổng số ngày giao dịch: {len(signals)}")
    print(f"Số tín hiệu Long: {(signals == 1).sum()}")
    print(f"Số tín hiệu Short: {(signals == -1).sum()}")
    print(f"Số tín hiệu Neutral: {(signals == 0).sum()}")
    
    # Xử lý từng ngày
    for i, (date, signal) in enumerate(signals.items()):
        vn05_price = vn05_prices.get(date, 0)
        futures_price = futures_prices.get(date, 0)
        z_val = z_score.get(date, 0)
        spread_val = spread.get(date, 0)
        
        # Tính HPR cho các vị thế hiện tại
        current_hpr = 0
        for pos_info in position_history:
            if pos_info['status'] == 'OPEN':
                hpr = calculate_hpr(vn05_prices, futures_prices, pos_info)
                current_hpr = hpr.get(date, 0)
                break
        
        if signal == 0:
            # Không có tín hiệu
            trading_history.append({
                'date': date,
                'signal': 0,
                'vn05_price': vn05_price,
                'futures_price': futures_price,
                'z_score': z_val,
                'spread': spread_val,
                'action': 'HOLD',
                'position_id': None,
                'reason': 'No signal',
                'hpr': current_hpr
            })
            continue
        
        # Xác định loại vị thế
        if signal > 0:
            position_type = PositionType.LONG_VN05_SHORT_FUTURES
            action_name = "LONG_VN05_SHORT_FUTURES"
        else:
            position_type = PositionType.SHORT_VN05_LONG_FUTURES
            action_name = "SHORT_VN05_LONG_FUTURES"
        
        # Kiểm tra có thể mở vị thế không
        can_open, reason = pm.can_open_position(position_type)
        
        if can_open and vn05_price > 0 and futures_price > 0:
            try:
                # Mở vị thế mới
                position = pm.open_position(
                    position_type=position_type,
                    vn05_price=vn05_price,
                    futures_price=futures_price,
                    hedge_ratio=1.2,
                    entry_date=date.strftime('%Y-%m-%d')
                )
                
                trading_history.append({
                    'date': date,
                    'signal': signal,
                    'vn05_price': vn05_price,
                    'futures_price': futures_price,
                    'z_score': z_val,
                    'spread': spread_val,
                    'action': f'OPEN_{action_name}',
                    'position_id': position['id'],
                    'reason': 'Position opened successfully',
                    'hpr': 0  # HPR = 0 khi mở vị thế
                })
                
                # Lưu thông tin vị thế
                pos_info = {
                    'date': date,
                    'position_id': position['id'],
                    'type': position_type.value,
                    'vn05_price': vn05_price,
                    'futures_price': futures_price,
                    'hedge_ratio': 1.2,
                    'vn05_quantity': position['vn05_quantity'],
                    'futures_quantity': position['futures_quantity'],
                    'allocated_capital': position['position_value'],
                    'direction_vn05': position['direction_vn05'],
                    'direction_futures': position['direction_futures'],
                    'z_score': z_val,
                    'spread': spread_val,
                    'status': 'OPEN'
                }
                position_history.append(pos_info)
                
                # Tính HPR cho vị thế mới
                hpr = calculate_hpr(vn05_prices, futures_prices, pos_info)
                hpr_data[position['id']] = hpr
                
            except ValueError as e:
                trading_history.append({
                    'date': date,
                    'signal': signal,
                    'vn05_price': vn05_price,
                    'futures_price': futures_price,
                    'z_score': z_val,
                    'spread': spread_val,
                    'action': f'FAILED_{action_name}',
                    'position_id': None,
                    'reason': str(e),
                    'hpr': current_hpr
                })
        else:
            trading_history.append({
                'date': date,
                'signal': signal,
                'vn05_price': vn05_price,
                'futures_price': futures_price,
                'z_score': z_val,
                'spread': spread_val,
                'action': f'REJECTED_{action_name}',
                'position_id': None,
                'reason': reason,
                'hpr': current_hpr
            })
    
    return trading_history, position_history, hpr_data, pm

def analyze_positions_with_hpr(trading_history, position_history, hpr_data, pm):
    """Phân tích vị thế với HPR"""
    print("\n" + "=" * 80)
    print("PHÂN TÍCH VỊ THẾ VỚI HPR")
    print("=" * 80)
    
    df = pd.DataFrame(trading_history)
    pos_df = pd.DataFrame(position_history)
    
    # Thống kê tổng quan
    print(f"\n1. THỐNG KÊ TỔNG QUAN:")
    print(f"   Tổng số ngày giao dịch: {len(df)}")
    print(f"   Số vị thế đã mở: {len(pos_df)}")
    print(f"   Vị thế đang mở: {len(pm.positions[PositionType.LONG_VN05_SHORT_FUTURES]) + len(pm.positions[PositionType.SHORT_VN05_LONG_FUTURES])}")
    print(f"   Vốn đã sử dụng: {pm.used_capital:,.0f} VND")
    print(f"   Vốn khả dụng: {pm.available_capital:,.0f} VND")
    
    # Phân tích HPR
    print(f"\n2. PHÂN TÍCH HPR:")
    if len(hpr_data) > 0:
        for pos_id, hpr_series in hpr_data.items():
            if len(hpr_series) > 0:
                print(f"   {pos_id}:")
                print(f"     HPR trung bình: {hpr_series.mean():.4f} ({hpr_series.mean()*100:.2f}%)")
                print(f"     HPR tối đa: {hpr_series.max():.4f} ({hpr_series.max()*100:.2f}%)")
                print(f"     HPR tối thiểu: {hpr_series.min():.4f} ({hpr_series.min()*100:.2f}%)")
                print(f"     HPR cuối cùng: {hpr_series.iloc[-1]:.4f} ({hpr_series.iloc[-1]*100:.2f}%)")
                print(f"     Volatility: {hpr_series.std():.4f} ({hpr_series.std()*100:.2f}%)")
    
    # Thống kê theo loại vị thế
    if len(pos_df) > 0:
        print(f"\n3. THỐNG KÊ THEO LOẠI VỊ THẾ:")
        type_stats = pos_df['type'].value_counts()
        for pos_type, count in type_stats.items():
            print(f"   {pos_type}: {count}")
    
    # Thống kê theo hành động
    print(f"\n4. THỐNG KÊ THEO HÀNH ĐỘNG:")
    action_stats = df['action'].value_counts()
    for action, count in action_stats.items():
        print(f"   {action}: {count}")
    
    # Hiển thị chi tiết các vị thế
    if len(pos_df) > 0:
        print(f"\n5. CHI TIẾT CÁC VỊ THẾ:")
        for _, pos in pos_df.iterrows():
            print(f"\n   Vị thế: {pos['position_id']}")
            print(f"   Ngày mở: {pos['date'].strftime('%Y-%m-%d')}")
            print(f"   Loại: {pos['type']}")
            print(f"   VN05: {pos['direction_vn05']} {pos['vn05_quantity']:.0f} shares @ {pos['vn05_price']:.2f}")
            print(f"   VN30F1M: {pos['direction_futures']} {pos['futures_quantity']:.0f} contracts @ {pos['futures_price']:.2f}")
            print(f"   Hedge ratio: {pos['hedge_ratio']:.2f}")
            print(f"   Z-score: {pos['z_score']:.3f}")
            print(f"   Spread: {pos['spread']:.2f}")
            print(f"   Vốn phân bổ: {pos['allocated_capital']:,.0f} VND")
            
            # Kiểm tra logic long/short
            if pos['type'] == 'long_vn05_short_futures':
                print(f"   ✅ Logic: Long VN05 (mua rổ cổ phiếu) + Short VN30F1M (bán futures)")
                print(f"   ✅ Kỳ vọng: VN05 tăng nhanh hơn VN30F1M")
            else:
                print(f"   ✅ Logic: Short VN05 (bán rổ cổ phiếu) + Long VN30F1M (mua futures)")
                print(f"   ✅ Kỳ vọng: VN05 giảm nhanh hơn VN30F1M")
    
    return df, pos_df

def create_hpr_plot(df, pos_df, hpr_data):
    """Tạo biểu đồ HPR"""
    plt.figure(figsize=(16, 12))
    
    # Subplot 1: Giá VN05 và VN30F1M
    plt.subplot(4, 1, 1)
    plt.plot(df['date'], df['vn05_price'], label='VN05 Price', color='blue', linewidth=2)
    plt.plot(df['date'], df['futures_price'], label='VN30F1M Price', color='orange', linewidth=2)
    plt.title('Giá VN05 và VN30F1M (Dữ liệu thực tế)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Spread
    plt.subplot(4, 1, 2)
    plt.plot(df['date'], df['spread'], label='Spread (VN05 - 1.2*VN30F1M)', color='green', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Spread giữa VN05 và VN30F1M')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Z-score và signals
    plt.subplot(4, 1, 3)
    plt.plot(df['date'], df['z_score'], label='Z-Score', color='purple', linewidth=2)
    plt.axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label='Short Threshold')
    plt.axhline(y=-1.5, color='g', linestyle='--', alpha=0.5, label='Long Threshold')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Highlight signals
    long_signals = df[df['signal'] == 1]
    short_signals = df[df['signal'] == -1]
    plt.scatter(long_signals['date'], long_signals['z_score'], 
                color='green', marker='^', s=100, label='Long Signal', zorder=5)
    plt.scatter(short_signals['date'], short_signals['z_score'], 
                color='red', marker='v', s=100, label='Short Signal', zorder=5)
    
    plt.title('Z-Score và Trading Signals')
    plt.ylabel('Z-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: HPR
    plt.subplot(4, 1, 4)
    if len(hpr_data) > 0:
        for pos_id, hpr_series in hpr_data.items():
            if len(hpr_series) > 0:
                plt.plot(hpr_series.index, hpr_series.values, 
                        label=f'HPR {pos_id}', linewidth=2, alpha=0.8)
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('HPR (Holding Period Return) qua thời gian')
        plt.ylabel('HPR')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Highlight position openings
        for _, pos in pos_df.iterrows():
            plt.axvline(x=pos['date'], color='blue', linestyle=':', alpha=0.7)
            plt.text(pos['date'], 0, pos['position_id'].split('_')[-1], 
                    rotation=90, fontsize=8, ha='right')
    else:
        plt.text(0.5, 0.5, 'Không có dữ liệu HPR', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('HPR (Holding Period Return) qua thời gian')
        plt.ylabel('HPR')
        plt.xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('result/plots/position_hpr_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_hpr_summary_table(pos_df, hpr_data):
    """Tạo bảng tóm tắt HPR"""
    if len(pos_df) == 0:
        print("\nKhông có vị thế nào để hiển thị")
        return
    
    print("\n" + "=" * 80)
    print("BẢNG TÓM TẮT HPR")
    print("=" * 80)
    
    # Tạo bảng tóm tắt
    summary_data = []
    for _, pos in pos_df.iterrows():
        pos_id = pos['position_id']
        hpr_series = hpr_data.get(pos_id, pd.Series())
        
        if len(hpr_series) > 0:
            summary_data.append({
                'Position ID': pos_id,
                'Entry Date': pos['date'].strftime('%Y-%m-%d'),
                'Type': pos['type'],
                'VN05 Price': f"{pos['vn05_price']:.2f}",
                'VN30F1M Price': f"{pos['futures_price']:.2f}",
                'Hedge Ratio': f"{pos['hedge_ratio']:.2f}",
                'Capital (M VND)': f"{pos['allocated_capital']/1_000_000:.0f}",
                'Z-Score': f"{pos['z_score']:.3f}",
                'Spread': f"{pos['spread']:.2f}",
                'HPR Mean (%)': f"{hpr_series.mean()*100:.2f}",
                'HPR Max (%)': f"{hpr_series.max()*100:.2f}",
                'HPR Min (%)': f"{hpr_series.min()*100:.2f}",
                'HPR Final (%)': f"{hpr_series.iloc[-1]*100:.2f}",
                'Volatility (%)': f"{hpr_series.std()*100:.2f}"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Lưu vào file
        summary_df.to_csv('result/position_hpr_summary.csv', index=False)
        print(f"\nĐã lưu bảng tóm tắt HPR vào: result/position_hpr_summary.csv")

def main():
    """Hàm chính"""
    print("Bắt đầu phân tích vị thế với HPR...")
    
    # Load dữ liệu thực tế
    futures_prices, vn30_stocks = load_real_data()
    if futures_prices is None or vn30_stocks is None:
        print("Không thể load dữ liệu thực tế")
        return
    
    # Tạo rổ VN05
    vn05_prices, selected_stocks = create_vn05_basket(vn30_stocks)
    
    # Tính spread và signals
    spread, z_score, signals = calculate_spread_and_signals(vn05_prices, futures_prices)
    
    # Mô phỏng giao dịch với HPR
    trading_history, position_history, hpr_data, pm = simulate_real_trading_with_hpr(
        vn05_prices, futures_prices, signals, z_score, spread)
    
    # Phân tích vị thế với HPR
    df, pos_df = analyze_positions_with_hpr(trading_history, position_history, hpr_data, pm)
    
    # Tạo biểu đồ HPR
    create_hpr_plot(df, pos_df, hpr_data)
    
    # Tạo bảng tóm tắt HPR
    create_hpr_summary_table(pos_df, hpr_data)
    
    print("\n" + "=" * 80)
    print("PHÂN TÍCH HPR HOÀN THÀNH")
    print("=" * 80)

if __name__ == "__main__":
    main()
