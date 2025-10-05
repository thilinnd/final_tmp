# Position Manager - Quản Lý Vị Thế Statistical Arbitrage

## Tổng Quan

Position Manager được thiết kế để quản lý vị thế trong chiến lược Statistical Arbitrage với các quy tắc nghiêm ngặt:

- **Tối đa 2 vị thế cùng chiều** cho mỗi loại (Long VN05 hoặc Short VN05)
- **Mỗi vị thế chiếm 40% vốn** (`position_size = 0.4`)
- **Tổng vốn tối đa: 80%** (2 vị thế × 40%)
- **Không cho phép vị thế ngược chiều** khi đã có vị thế cùng chiều

## Cấu Trúc

### 1. PositionType Enum
```python
class PositionType(Enum):
    LONG_VN05_SHORT_FUTURES = "long_vn05_short_futures"    # Long VN05 + Short VN30F1M
    SHORT_VN05_LONG_FUTURES = "short_vn05_long_futures"    # Short VN05 + Long VN30F1M
```

### 2. PositionManager Class

#### Khởi tạo
```python
pm = PositionManager(
    initial_capital=1_000_000_000,  # 1 tỷ VND
    position_size=0.4               # 40% vốn cho mỗi vị thế
)
```

#### Các phương thức chính

##### `can_open_position(position_type)`
Kiểm tra có thể mở vị thế mới không
```python
can_open, reason = pm.can_open_position(PositionType.LONG_VN05_SHORT_FUTURES)
if can_open:
    print("Có thể mở vị thế")
else:
    print(f"Không thể mở vị thế: {reason}")
```

##### `open_position(position_type, vn05_price, futures_price, hedge_ratio)`
Mở vị thế mới
```python
position = pm.open_position(
    position_type=PositionType.LONG_VN05_SHORT_FUTURES,
    vn05_price=1200,
    futures_price=1180,
    hedge_ratio=1.0,
    entry_date="2024-01-15"
)
```

##### `close_position(position_id)`
Đóng vị thế theo ID
```python
closed_position = pm.close_position("long_vn05_short_futures_1")
```

##### `close_all_positions()`
Đóng tất cả vị thế
```python
closed_positions = pm.close_all_positions()
```

##### `get_position_summary()`
Lấy tóm tắt vị thế hiện tại
```python
summary = pm.get_position_summary()
print(f"Tổng vị thế: {summary['total_open_positions']}")
print(f"Vốn sử dụng: {summary['capital_utilization']:.1%}")
```

##### `print_status()`
In trạng thái vị thế hiện tại
```python
pm.print_status()
```

## Quy Tắc Mở Vị Thế

### Trường hợp 1: Chưa có vị thế nào
```python
# ✅ Có thể mở:
# - Long VN05 + Short VN30F1M
# - Short VN05 + Long VN30F1M
```

### Trường hợp 2: Đang có 1 vị thế Long VN05
```python
# Đã có: Long VN05 + Short VN30F1M
# ✅ Có thể mở thêm: Long VN05 + Short VN30F1M (vị thế 2)
# ❌ Không được: Short VN05 + Long VN30F1M (ngược chiều)
```

### Trường hợp 3: Đã có 2 vị thế cùng chiều
```python
# Đã có: 2 vị thế Long VN05 + 2 vị thế Short VN30F1M
# ✅ Chỉ có thể: Đóng 1 vị thế (giảm xuống 1 vị thế)
# ❌ Không được: Mở thêm vị thế mới (đã đạt giới hạn)
```

## Tích Hợp Vào BacktestingEngine

### 1. Khởi tạo
```python
from backtesting import BacktestingEngine
from position_manager import PositionManager, PositionType

engine = BacktestingEngine(config)
# PositionManager đã được tích hợp sẵn
```

### 2. Sử dụng trong chiến lược
```python
# Xử lý tín hiệu giao dịch với Position Manager
returns_df = engine.process_trading_signals_with_position_manager(
    signals_df=signals,
    vn05_prices=vn05_prices,
    futures_prices=futures_prices,
    hedge_ratios=hedge_ratios
)
```

### 3. Theo dõi vị thế
```python
# Lấy tóm tắt vị thế
summary = engine.get_position_summary()

# In trạng thái
engine.print_position_status()

# Đóng tất cả vị thế
closed_positions = engine.close_all_positions()
```

## Ví Dụ Sử Dụng

### Demo cơ bản
```python
from position_manager import PositionManager, PositionType

# Khởi tạo
pm = PositionManager(1_000_000_000, 0.4)

# Mở vị thế 1
pos1 = pm.open_position(
    PositionType.LONG_VN05_SHORT_FUTURES,
    vn05_price=1200,
    futures_price=1180,
    hedge_ratio=1.0
)

# Mở vị thế 2 (cùng chiều)
pos2 = pm.open_position(
    PositionType.LONG_VN05_SHORT_FUTURES,
    vn05_price=1210,
    futures_price=1190,
    hedge_ratio=1.0
)

# Thử mở vị thế 3 (sẽ bị từ chối)
try:
    pos3 = pm.open_position(
        PositionType.LONG_VN05_SHORT_FUTURES,
        vn05_price=1220,
        futures_price=1200,
        hedge_ratio=1.0
    )
except ValueError as e:
    print(f"Lỗi: {e}")  # Đã đạt giới hạn 2 vị thế cùng chiều

# In trạng thái
pm.print_status()
```

### Chạy demo đầy đủ
```bash
python demo_position_manager.py
```

## Lợi Ích

1. **Kiểm soát rủi ro**: Giới hạn số vị thế và vốn sử dụng
2. **Tính đối xứng**: Luôn có cả Long và Short
3. **Linh hoạt**: Có thể tăng/giảm vị thế theo tín hiệu
4. **Quản lý vốn**: Tối đa 80% vốn, 20% dự phòng
5. **Tự động hóa**: Kiểm tra điều kiện và thực hiện giao dịch tự động

## Cấu Hình

### Tham số chính
- `initial_capital`: Vốn ban đầu (mặc định: 1,000,000,000 VND)
- `position_size`: Tỷ lệ vốn cho mỗi vị thế (mặc định: 0.4 = 40%)
- `max_positions_per_direction`: Tối đa vị thế cùng chiều (mặc định: 2)

### Tích hợp vào config
```json
{
    "initial_capital": 1000000000,
    "position_size": 0.4,
    "max_positions_per_direction": 2
}
```

## Lưu Ý

1. **Vị thế cùng chiều**: Chỉ được mở cùng loại (Long VN05 hoặc Short VN05)
2. **Vị thế ngược chiều**: Không được mở khi đã có vị thế cùng chiều
3. **Giới hạn vốn**: Tổng vốn sử dụng không vượt quá 80%
4. **Quản lý rủi ro**: Cần kết hợp với stop loss và max drawdown
5. **Tối ưu hóa**: Có thể điều chỉnh `position_size` theo chiến lược



