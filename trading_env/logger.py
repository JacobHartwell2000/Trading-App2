import logging
from datetime import datetime
from typing import List, Dict, Union, Optional

# Initialize activity log list
activity_log: List[dict] = []

def format_trading_analysis(symbol: str, 
                          prediction: float, 
                          confidence: float,
                          strategy_signals: Optional[Dict] = None,
                          final_signal: Optional[float] = None) -> str:
    """Format trading analysis into a readable message"""
    message = [f"\n📊 Trading Analysis for {symbol}:"]
    message.append(f"ML Prediction: {'BUY' if prediction == 1 else 'SELL'} (confidence: {confidence:.2f})")
    
    if strategy_signals:
        message.append("\nStrategy Signals:")
        for strategy, signal in strategy_signals.items():
            message.append(f"  • {strategy}: {'BUY' if signal > 0.5 else 'SELL'} ({signal:.2f})")
    
    if final_signal is not None:
        message.append(f"\nCombined Signal: {'BUY' if final_signal > 0.5 else 'SELL'} ({final_signal:.2f})")
    
    return "\n".join(message)

def format_trade_execution(symbol: str, details: Dict) -> str:
    """Format trade execution details into a readable message"""
    return f"""
✅ EXECUTING {details.get('side', 'BUY')} ORDER for {symbol}:
   Shares: {details.get('position_size', 0)}
   Price: ${details.get('price', 0):.2f}
   Stop Loss: ${details.get('stop_loss', 0):.2f}
   Take Profit: ${details.get('take_profit', 0):.2f}
"""

def format_no_trade_reasons(symbol: str, reasons: List[str]) -> str:
    """Format no-trade reasons into a readable message"""
    message = [f"\n❌ NO TRADE for {symbol}:"]
    for reason in reasons:
        message.append(f"   • {reason}")
    return "\n".join(message)

def log_activity(message: str, 
                entry_type: str = "info",
                trading_data: Optional[Dict] = None) -> None:
    """
    Log an activity with timestamp and optional trading data
    
    Parameters:
        message: The log message
        entry_type: Type of log entry (info, analysis, trade, no_trade)
        trading_data: Optional dictionary containing trading-related data
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format message based on entry type and trading data
    if entry_type == "analysis" and trading_data:
        formatted_message = format_trading_analysis(
            trading_data.get('symbol', ''),
            trading_data.get('prediction', 0),
            trading_data.get('confidence', 0),
            trading_data.get('strategy_signals'),
            trading_data.get('final_signal')
        )
    elif entry_type == "trade" and trading_data:
        formatted_message = format_trade_execution(
            trading_data.get('symbol', ''),
            trading_data
        )
    elif entry_type == "no_trade" and trading_data:
        formatted_message = format_no_trade_reasons(
            trading_data.get('symbol', ''),
            trading_data.get('reasons', [])
        )
    else:
        formatted_message = message
    
    # Create log entry
    log_entry = {
        'timestamp': timestamp,
        'type': entry_type,
        'message': formatted_message,
        'trading_data': trading_data
    }
    
    # Add to activity log
    activity_log.append(log_entry)
    
    # Keep only last 100 entries
    while len(activity_log) > 100:
        activity_log.pop(0)
    
    # Also print to console for debugging
    print(f"[{timestamp}] {formatted_message}")

def get_activity_log() -> List[dict]:
    """Get the current activity log"""
    return activity_log

def clear_activity_log() -> None:
    """Clear the activity log"""
    activity_log.clear() 