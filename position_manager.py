"""
Position Manager for Statistical Arbitrage Strategy
Quản lý vị thế theo quy tắc: tối đa 2 vị thế cùng chiều
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal


class PositionType(Enum):
    """Loại vị thế"""
    LONG_VN05_SHORT_FUTURES = "long_vn05_short_futures"
    SHORT_VN05_LONG_FUTURES = "short_vn05_long_futures"


class PositionManager:
    """
    Quản lý vị thế theo quy tắc:
    - Tối đa 2 vị thế cùng chiều cho mỗi loại
    - Mỗi vị thế chiếm 4% vốn
    - Tổng vốn tối đa: 16%
    """
    
    def __init__(self, initial_capital: float, position_size: float = 0.04):
        """
        Khởi tạo Position Manager
        
        Args:
            initial_capital (float): Vốn ban đầu
            position_size (float): Tỷ lệ vốn cho mỗi vị thế (mặc định 0.04 = 4%)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions_per_direction = 2
        
        # Theo dõi vị thế hiện tại
        self.positions: Dict[PositionType, List[Dict]] = {
            PositionType.LONG_VN05_SHORT_FUTURES: [],
            PositionType.SHORT_VN05_LONG_FUTURES: []
        }
        
        # Theo dõi vốn đã sử dụng
        self.used_capital = 0.0
        self.available_capital = initial_capital
        
    def can_open_position(self, position_type: PositionType) -> Tuple[bool, str]:
        """
        Kiểm tra có thể mở vị thế mới không
        
        Args:
            position_type (PositionType): Loại vị thế muốn mở
            
        Returns:
            Tuple[bool, str]: (có thể mở, lý do)
        """
        # Kiểm tra vốn khả dụng
        required_capital = self.initial_capital * self.position_size
        if self.available_capital < required_capital:
            return False, f"Không đủ vốn. Cần: {required_capital:,.0f}, Có: {self.available_capital:,.0f}"
        
        # Kiểm tra số lượng vị thế cùng chiều
        current_positions = self.positions[position_type]
        if len(current_positions) >= self.max_positions_per_direction:
            return False, f"Đã đạt giới hạn {self.max_positions_per_direction} vị thế cùng chiều"
        
        # Kiểm tra vị thế ngược chiều
        opposite_type = self._get_opposite_position_type(position_type)
        opposite_positions = self.positions[opposite_type]
        
        # Chỉ cấm vị thế ngược chiều khi đã có vị thế cùng chiều
        if len(opposite_positions) > 0 and len(current_positions) > 0:
            return False, f"Đã có vị thế cùng chiều. Chỉ được mở cùng chiều hoặc đóng hết để mở ngược chiều"
        
        # Cấm vị thế ngược chiều khi đã có vị thế ngược chiều
        if len(opposite_positions) > 0 and len(current_positions) == 0:
            return False, f"Đã có vị thế ngược chiều. Chỉ được mở cùng chiều hoặc đóng hết để mở ngược chiều"
        
        return True, "Có thể mở vị thế mới"
    
    def open_position(self, position_type: PositionType, 
                     vn05_price: float, futures_price: float,
                     hedge_ratio: float = 1.0,
                     entry_date: str = None) -> Dict:
        """
        Mở vị thế mới
        
        Args:
            position_type (PositionType): Loại vị thế
            vn05_price (float): Giá VN05
            futures_price (float): Giá VN30F1M
            hedge_ratio (float): Tỷ lệ hedge
            entry_date (str): Ngày mở vị thế
            
        Returns:
            Dict: Thông tin vị thế đã mở
        """
        # Kiểm tra có thể mở không
        can_open, reason = self.can_open_position(position_type)
        if not can_open:
            raise ValueError(f"Không thể mở vị thế: {reason}")
        
        # Tính toán vị thế
        position_value = self.initial_capital * self.position_size
        
        if position_type == PositionType.LONG_VN05_SHORT_FUTURES:
            # Long VN05, Short VN30F1M
            vn05_quantity = position_value / vn05_price
            futures_quantity = (position_value * hedge_ratio) / futures_price
            direction_vn05 = "LONG"
            direction_futures = "SHORT"
        else:
            # Short VN05, Long VN30F1M
            vn05_quantity = position_value / vn05_price
            futures_quantity = (position_value * hedge_ratio) / futures_price
            direction_vn05 = "SHORT"
            direction_futures = "LONG"
        
        # Tạo thông tin vị thế
        position_info = {
            'id': f"{position_type.value}_{len(self.positions[position_type]) + 1}",
            'type': position_type,
            'entry_date': entry_date or pd.Timestamp.now().strftime('%Y-%m-%d'),
            'vn05_price': vn05_price,
            'futures_price': futures_price,
            'hedge_ratio': hedge_ratio,
            'vn05_quantity': vn05_quantity,
            'futures_quantity': futures_quantity,
            'direction_vn05': direction_vn05,
            'direction_futures': direction_futures,
            'position_value': position_value,
            'status': 'OPEN'
        }
        
        # Thêm vào danh sách vị thế
        self.positions[position_type].append(position_info)
        
        # Cập nhật vốn
        self.used_capital += position_value
        self.available_capital -= position_value
        
        return position_info
    
    def close_position(self, position_id: str) -> Dict:
        """
        Đóng vị thế theo ID
        
        Args:
            position_id (str): ID của vị thế cần đóng
            
        Returns:
            Dict: Thông tin vị thế đã đóng
        """
        # Tìm vị thế trong tất cả loại
        for position_type, positions in self.positions.items():
            for i, position in enumerate(positions):
                if position['id'] == position_id and position['status'] == 'OPEN':
                    # Đóng vị thế
                    position['status'] = 'CLOSED'
                    position['close_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
                    
                    # Cập nhật vốn
                    self.used_capital -= position['position_value']
                    self.available_capital += position['position_value']
                    
                    return position
        
        raise ValueError(f"Không tìm thấy vị thế với ID: {position_id}")
    
    def close_all_positions(self) -> List[Dict]:
        """
        Đóng tất cả vị thế
        
        Returns:
            List[Dict]: Danh sách vị thế đã đóng
        """
        closed_positions = []
        
        for position_type, positions in self.positions.items():
            for position in positions:
                if position['status'] == 'OPEN':
                    position['status'] = 'CLOSED'
                    position['close_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
                    closed_positions.append(position)
        
        # Cập nhật vốn
        self.used_capital = 0.0
        self.available_capital = self.initial_capital
        
        return closed_positions
    
    def get_position_summary(self) -> Dict:
        """
        Lấy tóm tắt vị thế hiện tại
        
        Returns:
            Dict: Tóm tắt vị thế
        """
        total_positions = sum(len([p for p in positions if p['status'] == 'OPEN']) 
                             for positions in self.positions.values())
        
        long_vn05_count = len([p for p in self.positions[PositionType.LONG_VN05_SHORT_FUTURES] 
                              if p['status'] == 'OPEN'])
        short_vn05_count = len([p for p in self.positions[PositionType.SHORT_VN05_LONG_FUTURES] 
                               if p['status'] == 'OPEN'])
        
        return {
            'total_open_positions': total_positions,
            'long_vn05_positions': long_vn05_count,
            'short_vn05_positions': short_vn05_count,
            'used_capital': self.used_capital,
            'available_capital': self.available_capital,
            'capital_utilization': self.used_capital / self.initial_capital,
            'max_positions_reached': total_positions >= (self.max_positions_per_direction * 2)
        }
    
    def get_available_actions(self) -> List[str]:
        """
        Lấy danh sách hành động có thể thực hiện
        
        Returns:
            List[str]: Danh sách hành động
        """
        actions = []
        summary = self.get_position_summary()
        
        # Kiểm tra có thể mở vị thế mới không
        for position_type in PositionType:
            can_open, _ = self.can_open_position(position_type)
            if can_open:
                actions.append(f"OPEN_{position_type.value}")
        
        # Kiểm tra có thể đóng vị thế không
        for position_type, positions in self.positions.items():
            for position in positions:
                if position['status'] == 'OPEN':
                    actions.append(f"CLOSE_{position['id']}")
        
        return actions
    
    def _get_opposite_position_type(self, position_type: PositionType) -> PositionType:
        """
        Lấy loại vị thế ngược chiều
        
        Args:
            position_type (PositionType): Loại vị thế hiện tại
            
        Returns:
            PositionType: Loại vị thế ngược chiều
        """
        if position_type == PositionType.LONG_VN05_SHORT_FUTURES:
            return PositionType.SHORT_VN05_LONG_FUTURES
        else:
            return PositionType.LONG_VN05_SHORT_FUTURES
    
    def print_status(self):
        """In trạng thái vị thế hiện tại"""
        summary = self.get_position_summary()
        
        print("=" * 60)
        print("TRẠNG THÁI VỊ THẾ HIỆN TẠI")
        print("=" * 60)
        print(f"Tổng vị thế đang mở: {summary['total_open_positions']}")
        print(f"Long VN05: {summary['long_vn05_positions']}")
        print(f"Short VN05: {summary['short_vn05_positions']}")
        print(f"Vốn đã sử dụng: {summary['used_capital']:,.0f} VND")
        print(f"Vốn khả dụng: {summary['available_capital']:,.0f} VND")
        print(f"Tỷ lệ sử dụng vốn: {summary['capital_utilization']:.1%}")
        print(f"Đã đạt giới hạn: {'Có' if summary['max_positions_reached'] else 'Chưa'}")
        
        print("\nHÀNH ĐỘNG CÓ THỂ THỰC HIỆN:")
        actions = self.get_available_actions()
        for action in actions:
            print(f"  - {action}")
        
        print("=" * 60)


# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo Position Manager
    pm = PositionManager(initial_capital=10_000_000_000, position_size=0.04)
    
    # In trạng thái ban đầu
    pm.print_status()
    
    # Mở vị thế 1: Long VN05 + Short VN30F1M
    try:
        pos1 = pm.open_position(
            position_type=PositionType.LONG_VN05_SHORT_FUTURES,
            vn05_price=1200,
            futures_price=1180,
            hedge_ratio=1.0,
            entry_date="2024-01-15"
        )
        print(f"\nĐã mở vị thế: {pos1['id']}")
        pm.print_status()
    except ValueError as e:
        print(f"Lỗi: {e}")
    
    # Mở vị thế 2: Long VN05 + Short VN30F1M (cùng chiều)
    try:
        pos2 = pm.open_position(
            position_type=PositionType.LONG_VN05_SHORT_FUTURES,
            vn05_price=1210,
            futures_price=1190,
            hedge_ratio=1.0,
            entry_date="2024-01-16"
        )
        print(f"\nĐã mở vị thế: {pos2['id']}")
        pm.print_status()
    except ValueError as e:
        print(f"Lỗi: {e}")
    
    # Thử mở vị thế 3 (sẽ bị từ chối)
    try:
        pos3 = pm.open_position(
            position_type=PositionType.LONG_VN05_SHORT_FUTURES,
            vn05_price=1220,
            futures_price=1200,
            hedge_ratio=1.0,
            entry_date="2024-01-17"
        )
        print(f"\nĐã mở vị thế: {pos3['id']}")
    except ValueError as e:
        print(f"Lỗi: {e}")
    
    # Thử mở vị thế ngược chiều (sẽ bị từ chối)
    try:
        pos4 = pm.open_position(
            position_type=PositionType.SHORT_VN05_LONG_FUTURES,
            vn05_price=1190,
            futures_price=1210,
            hedge_ratio=1.0,
            entry_date="2024-01-18"
        )
        print(f"\nĐã mở vị thế: {pos4['id']}")
    except ValueError as e:
        print(f"Lỗi: {e}")
    
    # Đóng 1 vị thế
    try:
        closed_pos = pm.close_position(pos1['id'])
        print(f"\nĐã đóng vị thế: {closed_pos['id']}")
        pm.print_status()
    except ValueError as e:
        print(f"Lỗi: {e}")
