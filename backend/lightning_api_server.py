"""
Flask API Server for Enhanced Lightning GEX
Integrates with Alpaca for paper/live trading
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Enhanced Lightning GEX
from enhanced_lightning_system import EnhancedLightningGEX

# Alpaca API
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    print("⚠️  Alpaca SDK not installed. Install: pip install alpaca-py")
    ALPACA_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Initialize Enhanced Lightning GEX
print("\n" + "="*80)
print("INITIALIZING ENHANCED LIGHTNING GEX API SERVER")
print("="*80)

system = EnhancedLightningGEX(
    initial_capital=float(os.getenv('INITIAL_CAPITAL', 100000)),
    master_password=os.getenv('MASTER_PASSWORD', 'default_password'),
    enable_rl_sizing=os.getenv('ENABLE_RL_SIZING', 'true').lower() == 'true',
    enable_memory=os.getenv('ENABLE_MEMORY', 'true').lower() == 'true',
    enable_multi_agent=os.getenv('ENABLE_MULTI_AGENT', 'true').lower() == 'true',
    enable_hierarchical=os.getenv('ENABLE_HIERARCHICAL', 'true').lower() == 'true',
    enable_security=os.getenv('ENABLE_SECURITY', 'true').lower() == 'true'
)

# Initialize Alpaca clients
if ALPACA_AVAILABLE:
    alpaca_trading = TradingClient(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper=True  # Paper trading mode
    )
    
    alpaca_data = StockHistoricalDataClient(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY')
    )
    
    print("✓ Alpaca Paper Trading Connected")
    print(f"  Mode: {os.getenv('ALPACA_TRADING_MODE', 'PAPER')}")
else:
    alpaca_trading = None
    alpaca_data = None
    print("○ Alpaca Not Available (simulation mode)")

print("="*80 + "\n")


def calculate_technical_indicators(df):
    """Calculate all technical indicators needed"""
    
    # Moving averages
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(14).mean()
    
    # Stochastic
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stochastic'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    
    # ADX
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = ranges.max(axis=1)
    atr_14 = tr.rolling(14).mean()
    
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(14).mean()
    
    # Volume
    df['avg_volume'] = df['Volume'].rolling(20).mean()
    
    return df


def calculate_support_resistance(df):
    """Calculate support and resistance levels"""
    
    recent = df.tail(50)
    
    # Simple method: recent highs and lows
    resistance = recent['High'].max()
    support = recent['Low'].min()
    
    return support, resistance


def simulate_gex_data(ticker, price):
    """
    Simulate GEX/Charm/Vanna data
    In production, replace with real options data from CBOE/Tradier/etc.
    """
    
    # Placeholder: Use price momentum as proxy for GEX
    # This is SIMPLIFIED - you'll want real options flow data
    
    # For demo purposes, generate reasonable values
    import random
    
    gex_signal = random.uniform(-0.5, 0.8)
    charm_pressure = random.uniform(-0.3, 0.7)
    vanna_sensitivity = random.uniform(-0.2, 0.6)
    dark_pool_flow = random.uniform(-0.4, 0.7)
    
    return {
        'gex_signal': gex_signal,
        'charm_pressure': charm_pressure,
        'vanna_sensitivity': vanna_sensitivity,
        'dark_pool_flow': dark_pool_flow
    }


@app.route('/')
def home():
    """API home"""
    return jsonify({
        'service': 'Enhanced Lightning GEX API',
        'version': '1.0.0',
        'status': 'running',
        'alpaca_connected': ALPACA_AVAILABLE,
        'trading_mode': os.getenv('ALPACA_TRADING_MODE', 'PAPER'),
        'enhancements': {
            'rl_sizing': system.enable_rl_sizing,
            'memory': system.enable_memory,
            'multi_agent': system.enable_multi_agent,
            'hierarchical': system.enable_hierarchical,
            'security': system.enable_security
        }
    })


@app.route('/api/fetch_data', methods=['GET'])
def fetch_data():
    """Fetch market data for ticker"""
    
    ticker = request.args.get('ticker', 'SPY')
    
    try:
        # Fetch data using yfinance
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        
        if hist.empty:
            return jsonify({'error': f'No data for {ticker}'}), 404
        
        # Calculate indicators
        hist = calculate_technical_indicators(hist)
        
        # Get support/resistance
        support, resistance = calculate_support_resistance(hist)
        
        # Get latest values
        latest = hist.iloc[-1]
        
        # Get market data (SPY for market context)
        spy = yf.Ticker('SPY')
        spy_hist = spy.history(period="3mo")
        spy_hist = calculate_technical_indicators(spy_hist)
        spy_latest = spy_hist.iloc[-1]
        
        # Get VIX
        vix = yf.Ticker('^VIX')
        vix_hist = vix.history(period="1mo")
        vix_current = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 20.0
        vix_avg = vix_hist['Close'].mean() if not vix_hist.empty else 20.0
        
        # Simulate GEX data (replace with real data in production)
        options_data = simulate_gex_data(ticker, latest['Close'])
        
        # Prepare response
        data = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            
            # Price data
            'price': float(latest['Close']),
            'volume': int(latest['Volume']),
            'sma_20': float(latest['sma_20']),
            'sma_50': float(latest['sma_50']),
            'sma_200': float(latest['sma_200']),
            'rsi': float(latest['rsi']),
            'bb_upper': float(latest['bb_upper']),
            'bb_lower': float(latest['bb_lower']),
            'bb_width': float(latest['bb_width']),
            'stochastic': float(latest['stochastic']),
            'atr': float(latest['atr']),
            'adx': float(latest['adx']),
            'avg_volume': float(latest['avg_volume']),
            
            # Support/Resistance
            'resistance': float(resistance),
            'support': float(support),
            
            # Rolling averages for BB width and ATR
            'bb_width_avg': float(hist['bb_width'].tail(20).mean()),
            'atr_20_avg': float(hist['atr'].tail(20).mean()),
            'volatility': float(hist['Close'].pct_change().std() * np.sqrt(252)),
            
            # Market data
            'spy_trend': float((spy_latest['Close'] - spy_latest['sma_50']) / spy_latest['sma_50']),
            'vix': float(vix_current),
            'vix_20_avg': float(vix_avg),
            'market_adx': float(spy_latest['adx']),
            
            # Advance/Decline (simplified)
            'advance_decline': 1.2,  # Placeholder
            
            # Options data (simulated)
            **options_data
        }
        
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze_enhanced', methods=['POST'])
def analyze_enhanced():
    """Run enhanced Lightning GEX analysis"""
    
    try:
        data = request.json
        
        # Run analysis
        decision = system.analyze_ticker(
            ticker=data['ticker'],
            price_data=data['price_data'],
            options_data=data['options_data'],
            market_data=data['market_data']
        )
        
        return jsonify(decision)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/place_order', methods=['POST'])
def place_order():
    """Place order with Alpaca"""
    
    try:
        order_data = request.json
        
        if not ALPACA_AVAILABLE or alpaca_trading is None:
            return jsonify({
                'status': 'simulated',
                'message': 'Alpaca not available - order simulated',
                'order': order_data
            })
        
        # Prepare order
        if order_data['action'] == 'BUY':
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL
        
        # Create limit order
        order_request = LimitOrderRequest(
            symbol=order_data['ticker'],
            qty=order_data['quantity'],
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=order_data['limit_price']
        )
        
        # Submit order
        order = alpaca_trading.submit_order(order_request)
        
        return jsonify({
            'status': 'success',
            'order_id': order.id,
            'symbol': order.symbol,
            'qty': order.qty,
            'side': order.side.value,
            'type': order.type.value,
            'limit_price': order.limit_price,
            'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_account', methods=['GET'])
def get_account():
    """Get Alpaca account info"""
    
    try:
        if not ALPACA_AVAILABLE or alpaca_trading is None:
            return jsonify({'error': 'Alpaca not available'}), 503
        
        account = alpaca_trading.get_account()
        
        return jsonify({
            'account_number': account.account_number,
            'status': account.status.value,
            'currency': account.currency,
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'buying_power': float(account.buying_power),
            'pattern_day_trader': account.pattern_day_trader,
            'daytrade_count': account.daytrade_count,
            'trading_blocked': account.trading_blocked,
            'account_blocked': account.account_blocked
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_positions', methods=['GET'])
def get_positions():
    """Get current positions"""
    
    try:
        if not ALPACA_AVAILABLE or alpaca_trading is None:
            return jsonify([])
        
        positions = alpaca_trading.get_all_positions()
        
        result = []
        for pos in positions:
            result.append({
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'side': pos.side.value,
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'cost_basis': float(pos.cost_basis)
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/close_position', methods=['POST'])
def close_position():
    """Close a position"""
    
    try:
        data = request.json
        symbol = data['symbol']
        
        if not ALPACA_AVAILABLE or alpaca_trading is None:
            return jsonify({
                'status': 'simulated',
                'message': f'Position close simulated for {symbol}'
            })
        
        # Close position
        alpaca_trading.close_position(symbol)
        
        return jsonify({
            'status': 'success',
            'message': f'Position closed for {symbol}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/system_status', methods=['GET'])
def system_status():
    """Get system health status"""
    
    try:
        summary = system.get_performance_summary()
        
        # Add Alpaca account info if available
        if ALPACA_AVAILABLE and alpaca_trading:
            try:
                account = alpaca_trading.get_account()
                summary['alpaca'] = {
                    'connected': True,
                    'mode': 'PAPER',
                    'cash': float(account.cash),
                    'equity': float(account.equity),
                    'buying_power': float(account.buying_power)
                }
            except:
                summary['alpaca'] = {'connected': False}
        else:
            summary['alpaca'] = {'connected': False}
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/update_outcome', methods=['POST'])
def update_outcome():
    """Update trade outcome after exit"""
    
    try:
        data = request.json
        
        system.update_trade_outcome(
            trade_id=data['trade_id'],
            exit_price=data['exit_price'],
            exit_reason=data['exit_reason']
        )
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Run Flask app
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    
    print(f"\n🚀 Enhanced Lightning GEX API Server")
    print(f"   Listening on: http://0.0.0.0:{port}")
    print(f"   Mode: {os.getenv('ALPACA_TRADING_MODE', 'PAPER')}")
    print(f"   Alpaca: {'✓ Connected' if ALPACA_AVAILABLE else '○ Not Available'}")
    print(f"\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
