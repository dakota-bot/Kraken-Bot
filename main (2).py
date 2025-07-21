import time
import requests
import os
import hmac
import hashlib
import base64
import urllib.parse
import pandas as pd
import asyncio
import aiohttp
from datetime import datetime, timedelta

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
BASE_URL = "https://api.kraken.com"

WATCHLIST = [
    "XBTUSD", "ETHUSD", "ADAUSD", "SOLUSD", "AVAXUSD", "DOTUSD", "LTCUSD",
    "PEPEUSD", "WIFUSD", "BONKUSD", "FLOKIUSD", "SHIBUSD", "ARBUSD", "OPUSD",
    "INJUSD", "LDOUSD", "MKRUSD", "FETUSD", "APTUSD", "FILUSD", "PYTHUSD",
    "NEARUSD", "LINKUSD", "DOGEUSD", "SUIUSD", "AAVEUSD", "RUNEUSD", "SNXUSD",
    "TRXUSD", "UNIUSD", "ETCUSD", "BCHUSD", "ATOMUSD", "HBARUSD", "MATICUSD",
    "ALGOUSD", "CRVUSD", "COMPUSD", "XLMUSD", "POLUSD"
]

# OPTIMIZED FOR $50 DAILY PROFIT WITH SPREAD AWARENESS
SCAN_INTERVAL = 3.0    # Balanced scanning for quality opportunities
RSI_THRESHOLD = 25     # Slightly higher for better quality entries
TAKE_PROFIT = 0.025    # 2.5% profit target (more realistic after spreads)
TRAILING_STOP = 0.015  # 1.5% trailing stop 
TRIGGER_TRAIL_AT = 0.018  # Trigger trailing at 1.8% profit
TRADE_FRACTION = 0.50  # Use 50% of balance for fewer, larger trades
MIN_USD_VOLUME = 25.0  # $25 minimum to reduce spread impact
MAX_SPREAD = 0.003     # Maximum 0.3% spread allowed
MAX_POSITIONS = 2      # Fewer concurrent positions for better management
positions = {}

# Daily profit tracking
daily_profit = 0.0
daily_target = 50.0
session_start = time.time()

# Global nonce counter and API call tracking
_last_nonce = 0
_last_api_call = 0
_api_call_count = 0

def kraken_request(uri_path, data=None):
    global _last_nonce, _last_api_call, _api_call_count
    try:
        if data is None:
            data = {}

        # API rate limiting
        current_time = time.time()
        time_since_last = current_time - _last_api_call
        if time_since_last < 2.0:
            time.sleep(2.0 - time_since_last)

        # Generate unique nonce
        current_nonce = int(time.time() * 1000000000)
        if current_nonce <= _last_nonce:
            current_nonce = _last_nonce + 1000
        _last_nonce = current_nonce

        data['nonce'] = str(current_nonce)
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = uri_path.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(base64.b64decode(API_SECRET), message, hashlib.sha512)
        sigdigest = base64.b64encode(signature.digest())
        headers = {'API-Key': API_KEY, 'API-Sign': sigdigest.decode()}

        _last_api_call = time.time()
        _api_call_count += 1

        response = requests.post(BASE_URL + uri_path, headers=headers, data=data, timeout=10)
        result = response.json()

        # Enhanced error handling
        if result and 'error' in result:
            errors = result['error']
            if any('nonce' in str(err).lower() for err in errors):
                print(f"🔄 Nonce error #{_api_call_count}, retrying...")
                time.sleep(2.0)
                current_nonce = int(time.time() * 1000000000) + 10000
                _last_nonce = current_nonce
                data['nonce'] = str(current_nonce)
                postdata = urllib.parse.urlencode(data)
                encoded = (str(data['nonce']) + postdata).encode()
                message = uri_path.encode() + hashlib.sha256(encoded).digest()
                signature = hmac.new(base64.b64decode(API_SECRET), message, hashlib.sha512)
                sigdigest = base64.b64encode(signature.digest())
                headers = {'API-Key': API_KEY, 'API-Sign': sigdigest.decode()}
                response = requests.post(BASE_URL + uri_path, headers=headers, data=data, timeout=10)
                result = response.json()

            if result and 'error' in result and result['error']:
                print(f"❌ API Error: {result['error']}")

        return result
    except Exception as e:
        print(f"❌ API Exception: {e}")
        return None

async def fetch_ohlc_data(session, pair):
    try:
        async with session.get(f"{BASE_URL}/0/public/OHLC?pair={pair}&interval=1") as resp:
            data = await resp.json()
            if 'result' in data:
                ohlc_data = list(data['result'].values())[0]
                return pair, [float(candle[4]) for candle in ohlc_data[-50:]]
    except Exception as e:
        print(f"❌ OHLC error {pair}: {e}")
    return pair, []

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    df = pd.DataFrame({'close': prices})
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 1)

def calculate_trend_strength(prices):
    """Calculate trend strength using multiple indicators"""
    if len(prices) < 20:
        return 0

    # Simple trend calculation
    recent_avg = sum(prices[-5:]) / 5
    older_avg = sum(prices[-20:-15]) / 5
    trend = (recent_avg - older_avg) / older_avg
    return trend

def fetch_price_and_spread(pair):
    try:
        response = requests.get(f"{BASE_URL}/0/public/Ticker?pair={pair}")
        data = response.json()
        if 'result' in data:
            ticker_data = list(data['result'].values())[0]
            bid = float(ticker_data['b'][0])  # Best bid
            ask = float(ticker_data['a'][0])  # Best ask
            price = float(ticker_data['c'][0])  # Last price
            spread = (ask - bid) / price if price > 0 else 1.0
            return price, spread, bid, ask
    except Exception as e:
        print(f"❌ Price fetch error {pair}: {e}")
    return None, 1.0, None, None

def fetch_price(pair):
    price, _, _, _ = fetch_price_and_spread(pair)
    return price

def get_balance():
    result = kraken_request('/0/private/Balance')
    if result and 'result' in result:
        balances = result['result']
        usd_balance = float(balances.get('ZUSD', 0))
        print(f"💰 USD Balance: ${usd_balance:.2f}")
        return usd_balance
    return 0

def place_buy_order(pair, volume):
    data = {'pair': pair, 'type': 'buy', 'ordertype': 'market', 'volume': str(volume)}
    return kraken_request('/0/private/AddOrder', data)

def place_sell_order(pair, volume):
    data = {'pair': pair, 'type': 'sell', 'ordertype': 'market', 'volume': str(volume)}
    return kraken_request('/0/private/AddOrder', data)

def get_position_balance(pair):
    result = kraken_request('/0/private/Balance')
    if result and 'result' in result:
        balances = result['result']

        # Enhanced asset mapping
        asset_map = {
            'XBTUSD': 'XXBT', 'ETHUSD': 'XETH', 'LTCUSD': 'XLTC', 'DOGEUSD': 'XXDG',
            'ADAUSD': 'ADA', 'SOLUSD': 'SOL', 'AVAXUSD': 'AVAX', 'DOTUSD': 'DOT',
            'OPUSD': 'OP', 'INJUSD': 'INJ', 'APTUSD': 'APT', 'ARBUSD': 'ARB',
            'ALGOUSD': 'ALGO', 'ATOMUSD': 'ATOM', 'SNXUSD': 'SNX', 'LINKUSD': 'LINK',
            'SUIUSD': 'SUI', 'NEARUSD': 'NEAR', 'COMPUSD': 'COMP', 'CRVUSD': 'CRV',
            'AAVEUSD': 'AAVE', 'MKRUSD': 'MKR', 'UNIUSD': 'UNI', 'FETUSD': 'FET',
            'FILUSD': 'FIL', 'PYTHUSD': 'PYTH', 'RUNEUSD': 'RUNE', 'TRXUSD': 'TRX',
            'ETCUSD': 'ETC', 'BCHUSD': 'BCH', 'HBARUSD': 'HBAR', 'MATICUSD': 'MATIC',
            'XLMUSD': 'XLM', 'LDOUSD': 'LDO', 'BONKUSD': 'BONK', 'WIFUSD': 'WIF',
            'PEPEUSD': 'PEPE', 'FLOKIUSD': 'FLOKI', 'SHIBUSD': 'SHIB', 'POLUSD': 'POL'
        }

        asset = asset_map.get(pair, pair.replace('USD', ''))
        balance = float(balances.get(asset, 0))
        return balance
    return 0

def handle_position(pair, price, prices):
    global daily_profit

    if pair in positions:
        entry = positions[pair]
        entry_price = entry['buy_price']
        peak = max(entry['peak'], price)
        positions[pair]['peak'] = peak
        profit_pct = (price - entry_price) / entry_price
        time_held = time.time() - entry.get('timestamp', time.time())

        # Calculate trend to avoid selling into strong uptrends
        trend_strength = calculate_trend_strength(prices)

        should_sell = False
        sell_reason = ""

        # Enhanced selling logic for $50 daily target
        if time_held > 1800:  # 30 minute max hold
            should_sell = True
            sell_reason = f"⏰ 30MIN AUTO-SELL"
        elif profit_pct >= TAKE_PROFIT:  # 4% profit target
            should_sell = True
            sell_reason = f"💰 PROFIT TARGET +{profit_pct*100:.2f}%"
        elif profit_pct >= TRIGGER_TRAIL_AT and price < peak * (1 - TRAILING_STOP):
            # Don't trail if strong uptrend continues
            if trend_strength < 0.02:  # Only trail if trend is weakening
                should_sell = True
                sell_reason = f"📈 TRAILING STOP +{profit_pct*100:.2f}%"
        elif profit_pct <= -0.03:  # 3% stop loss
            should_sell = True
            sell_reason = f"🔴 STOP LOSS {profit_pct*100:.2f}%"
        elif time_held > 600 and profit_pct >= 0.015:  # Take 1.5% profit after 10 minutes
            should_sell = True
            sell_reason = f"🎯 QUICK PROFIT +{profit_pct*100:.2f}%"

        if should_sell:
            volume = get_position_balance(pair)
            if volume > 0:
                result = place_sell_order(pair, volume)
                if result and 'result' in result:
                    trade_profit = (price - entry_price) * volume
                    daily_profit += trade_profit

                    print(f"✅ SOLD {pair}: {sell_reason} | Profit: ${trade_profit:.2f} | Daily: ${daily_profit:.2f}/{daily_target}")
                    del positions[pair]
                    return True
                else:
                    print(f"❌ SELL ERROR {pair}: {result}")
    return False

def calculate_position_size(balance, price, pair):
    """Calculate optimal position size for $50 daily target"""
    # Target $2-4 profit per trade (need 12-25 winning trades for $50)
    target_profit_per_trade = 3.0  # $3 target per trade
    position_value = target_profit_per_trade / TAKE_PROFIT  # $75 position for $3 profit at 4%

    # Don't risk more than 30% of balance per trade
    max_position = balance * TRADE_FRACTION
    position_value = min(position_value, max_position)

    return max(position_value, MIN_USD_VOLUME)

async def trade_all():
    global daily_profit, session_start
    print(f"🚀 $50 DAILY PROFIT BOT - Target: ${daily_target}/day")
    print(f"📊 Strategy: 4% profit target, 3% stop loss, RSI < {RSI_THRESHOLD}")

    trade_count = 0
    last_balance_check = 0
    cached_usd_balance = 0.0

    while True:
        # Check daily profit progress
        hours_running = (time.time() - session_start) / 3600
        if hours_running > 0:
            hourly_rate = daily_profit / hours_running
            projected_daily = hourly_rate * 24
            print(f"📈 Daily Progress: ${daily_profit:.2f} | Projected: ${projected_daily:.2f} | Rate: ${hourly_rate:.2f}/hr")

        # Stop trading if daily target reached
        if daily_profit >= daily_target:
            print(f"🎯 DAILY TARGET REACHED! Profit: ${daily_profit:.2f}")
            await asyncio.sleep(60)  # Wait before continuing
            continue

        # Balance management
        current_time = time.time()
        if current_time - last_balance_check > 180:  # Check balance every 3 minutes
            cached_usd_balance = get_balance()
            last_balance_check = current_time

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_ohlc_data(session, pair) for pair in WATCHLIST]
            results = await asyncio.gather(*tasks)

        best_opportunities = []

        for pair, prices in results:
            if not prices:
                continue

            rsi = calculate_rsi(prices)
            price, spread, bid, ask = fetch_price_and_spread(pair)
            if not price:
                continue

            # Show market overview every 30 scans
            if trade_count % 30 == 0:
                spread_pct = spread * 100
                print(f"📊 {pair}: RSI {rsi:.1f} | ${price:.4f} | Spread: {spread_pct:.2f}%")

            # Handle existing positions
            handle_position(pair, price, prices)

            # Look for new opportunities - only with good RSI AND low spreads
            if (rsi < RSI_THRESHOLD and 
                pair not in positions and 
                len(positions) < MAX_POSITIONS and
                cached_usd_balance > MIN_USD_VOLUME and
                spread < MAX_SPREAD):  # NEW: Only trade if spread is acceptable

                trend_strength = calculate_trend_strength(prices)

                # Score the opportunity (lower spread = higher score)
                spread_bonus = (MAX_SPREAD - spread) * 1000  # Bonus for tighter spreads
                opportunity_score = (30 - rsi) + spread_bonus + (trend_strength * 100 if trend_strength > 0 else 0)
                best_opportunities.append((pair, rsi, price, opportunity_score, prices, spread))

        # Trade only the best opportunity
        if best_opportunities:
            best_opportunities.sort(key=lambda x: x[3], reverse=True)
            pair, rsi, price, score, prices, spread = best_opportunities[0]

            if score > 15:  # Higher threshold for quality
                position_size = calculate_position_size(cached_usd_balance, price, pair)
                spread_pct = spread * 100

                # Check minimum volumes
                min_volumes = {
                    'XBTUSD': 0.0002, 'ETHUSD': 0.005, 'BONKUSD': 500000, 'LTCUSD': 0.1,
                    'SOLUSD': 0.1, 'AVAXUSD': 0.1, 'APTUSD': 0.5, 'LINKUSD': 0.5,
                    'INJUSD': 0.1, 'NEARUSD': 1, 'SUIUSD': 1, 'BCHUSD': 0.005,
                    'AAVEUSD': 0.01, 'MKRUSD': 0.005, 'COMPUSD': 0.01,
                    'ADAUSD': 10, 'DOTUSD': 1, 'PEPEUSD': 10000000, 'WIFUSD': 5,
                    'FLOKIUSD': 100000, 'SHIBUSD': 10000000, 'ARBUSD': 2, 'OPUSD': 1,
                    'LDOUSD': 2, 'FETUSD': 2, 'FILUSD': 1, 'PYTHUSD': 10,
                    'DOGEUSD': 10, 'RUNEUSD': 1, 'SNXUSD': 2, 'TRXUSD': 10, 'UNIUSD': 0.5,
                    'ETCUSD': 0.1, 'ATOMUSD': 1, 'HBARUSD': 100, 'MATICUSD': 10,
                    'ALGOUSD': 10, 'CRVUSD': 2, 'XLMUSD': 10, 'POLUSD': 1
                }

                min_volume = min_volumes.get(pair, 0.001)
                volume = max(min_volume, round(position_size / price, 8))
                actual_allocation = volume * price
                expected_profit = actual_allocation * TAKE_PROFIT

                if actual_allocation <= cached_usd_balance and expected_profit >= 2.0:  # Higher minimum profit
                    print(f"🎯 BUYING {pair}: RSI {rsi:.1f} | Spread {spread_pct:.2f}% | ${actual_allocation:.2f} → ${expected_profit:.2f} profit target")

                    result = place_buy_order(pair, volume)
                    if result and 'result' in result:
                        positions[pair] = {
                            "buy_price": price, 
                            "peak": price, 
                            "timestamp": time.time(),
                            "target_profit": expected_profit,
                            "entry_spread": spread
                        }
                        trade_count += 1
                        cached_usd_balance -= actual_allocation
                        print(f"🟢 #{trade_count} BOUGHT {pair} @ ${price:.4f} | Target: ${expected_profit:.2f} | Spread: {spread_pct:.2f}%")
                    elif result and 'error' in result:
                        print(f"❌ BUY ERROR {pair}: {result['error']}")
                        if 'Insufficient' in str(result['error']):
                            cached_usd_balance = get_balance()
                else:
                    print(f"⏭️ SKIPPING {pair}: Expected profit ${expected_profit:.2f} too low or spread {spread_pct:.2f}% too high")

        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(trade_all())
    except KeyboardInterrupt:
        print("🛑 Bot stopped by user.")
        print(f"💰 Session profit: ${daily_profit:.2f}")