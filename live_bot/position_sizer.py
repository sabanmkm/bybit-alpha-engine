"""
POSITION SIZER - BYBIT MINIMUM NOTIONAL ADAPTIVE ENGINE
Sizes positions to exact exchange minimums with 10x leverage margin math.
"""
import math
import logging

logger = logging.getLogger("BybitBot")

def calculate_position_size(symbol: str, entry_price: float, stop_loss: float, available_margin: float, instrument_info: dict, leverage: int = 10):
    """
    Sizes to Bybit's EXACT minimum allowed order quantity.
    Ensures order >= minNotional ($5) and >= minOrderQty.
    """
    try:
        min_qty = float(instrument_info.get('minOrderQty', 0.1))
        qty_step = float(instrument_info.get('qtyStep', 0.1))
        min_notional = float(instrument_info.get('minNotional', 5.0))
        if min_notional <= 0:
            min_notional = 5.0

        # Calculate required quantity to meet minNotional floor
        notional_at_min_qty = min_qty * entry_price
        if notional_at_min_qty < min_notional:
            needed_qty = min_notional / entry_price
            steps = math.ceil(needed_qty / qty_step)
            order_qty = round(steps * qty_step, 6)
        else:
            order_qty = min_qty

        # Clean precision based on qty_step
        if qty_step >= 1:
            order_qty = int(order_qty)
        else:
            decimals = len(str(qty_step).split('.')[1]) if '.' in str(qty_step) else 2
            order_qty = round(order_qty, decimals)

        actual_notional = order_qty * entry_price
        margin_required = actual_notional / leverage

        # Hard guard for ETH: max $3.00 margin per trade
        if "ETH" in symbol and margin_required > 3.00:
            logger.warning(f"[{symbol}] Margin ${margin_required:.2f} exceeds micro-cap limit. Skipping.")
            return 0.0, 0.0, 0.0

        # Margin availability check
        if margin_required > available_margin:
            logger.warning(f"[{symbol}] Margin required (${margin_required:.2f}) > Available (${available_margin:.2f}). Skipping.")
            return 0.0, 0.0, 0.0

        risk_at_sl = order_qty * abs(entry_price - stop_loss)
        return order_qty, margin_required, risk_at_sl

    except Exception as e:
        logger.error(f"Error sizing position for {symbol}: {e}")
        return 0.0, 0.0, 0.0

def can_open_position(symbol: str, current_positions: list, margin_needed: float, available_balance: float) -> bool:
    """Safety check before order dispatch."""
    for pos in current_positions:
        if pos.get('symbol') == symbol and float(pos.get('size', 0)) > 0:
            logger.info(f"[{symbol}] Existing position active. Skipping duplicate entry.")
            return False
            
    if margin_needed > (available_balance * 0.90):  # Keep 10% cash buffer
        return False
        
    return True
