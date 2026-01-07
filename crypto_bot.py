import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import os

import logging
from logging.handlers import TimedRotatingFileHandler
from ccxt.base.errors import RequestTimeout, NetworkError, ExchangeNotAvailable, DDoSProtection

# =========================================================
# 0) ENV / WEBHOOK
# =========================================================
load_dotenv()
GOOGLE_CHAT_WEBHOOK = os.getenv("GOOGLE_CHAT_WEBHOOK")

# =========================================================
# 0.5) LOGGING (console + logs/bot.log + logs/bot_error.log)
# =========================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("crypto_bot")
logger.setLevel(logging.INFO)

fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)

# bot.log (rotate daily)
fh = TimedRotatingFileHandler(
    filename=os.path.join(LOG_DIR, "bot.log"),
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)

# bot_error.log (rotate daily)
eh = TimedRotatingFileHandler(
    filename=os.path.join(LOG_DIR, "bot_error.log"),
    when="midnight",
    interval=1,
    backupCount=14,
    encoding="utf-8"
)
eh.setLevel(logging.ERROR)
eh.setFormatter(fmt)

# Avoid duplicate handlers
if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(eh)


def send_google_chat(msg: str):
    if not GOOGLE_CHAT_WEBHOOK:
        logger.warning("Missing GOOGLE_CHAT_WEBHOOK in .env (alerts will not be sent)")
        return
    try:
        r = requests.post(GOOGLE_CHAT_WEBHOOK, json={"text": msg}, timeout=10)
        if r.status_code >= 400:
            logger.error(f"Google Chat webhook failed: {r.status_code} {r.text[:200]}")
        else:
            logger.info("Alert sent to Google Chat")
    except Exception as e:
        logger.exception(f"Error sending message to Google Chat: {e}")


# =========================================================
# 1) CONFIG (Crypto only | Long+Short | No leverage | Trading capital split)
# =========================================================
CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "WLD/USDT"]
TIMEFRAME = "1h"
SCAN_SEC = 600  # 10 minutes

# --- Portfolio split ---
HODL_CAPITAL = 0
TRADING_CAPITAL = 1500.0

# --- Monthly topup info (FYI / you can manually update TRADING_CAPITAL) ---
MONTHLY_TOPUP = 300.0

# --- Trade sizing based on TRADING_CAPITAL ---
BASE_TRADE_CAPITAL = 1000.0
TRADE_PCT = 0.60
TRADE_CAP_PCT = 0.80

# --- Direction ---
ALLOW_LONG = True
ALLOW_SHORT = True

# --- Risk / exits (ATR-based) ---
SL_ATR = 1.5
TP1_ATR = 1.5   # partial
TP2_ATR = 3.0   # final
MOVE_SL_TO_BE_ON_TP1 = True

# --- Filters (avoid dead/too wild markets) ---
MIN_ATR_PCT = 0.30
MAX_ATR_PCT = 2.50

# RSI filters
LONG_RSI_MIN, LONG_RSI_MAX = 40, 65
SHORT_RSI_MIN, SHORT_RSI_MAX = 35, 60

# --- Max concurrent trades ---
def max_open_trades(trading_cap: float) -> int:
    return 1 if trading_cap < 1000 else 2

# --- Cooldown (avoid spam) ---
COOLDOWN_SEC = 60 * 60 * 2  # 2 hours per event per symbol


# =========================================================
# 2) EXCHANGE (OKX spot)
# =========================================================
exchange = ccxt.okx({
    "enableRateLimit": True,
    "timeout": 30000,  # 30s
    "options": {
        "defaultType": "spot",
        # บางเวอร์ชันมี key นี้ ช่วยกันหลุดไป futures
        "fetchMarkets": "spot",
    },
})

# =========================================================
# 2.1) CCXT RETRY (generic)
# =========================================================
def ccxt_call_with_retry(fn, *args, retries: int = 5, base_sleep: float = 2.0, **kwargs):
    """
    Retry network-ish errors with exponential backoff.
    Returns: result or raises last exception
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except (RequestTimeout, NetworkError, ExchangeNotAvailable, DDoSProtection) as e:
            last_exc = e
            logger.warning(f"CCXT RETRY {attempt}/{retries} | {type(e).__name__}: {e}")
            sleep_s = base_sleep * (2 ** (attempt - 1))
            time.sleep(min(sleep_s, 30))  # cap 30s
        except Exception:
            raise
    raise last_exc


# =========================================================
# 3) STATE (in-memory)
# =========================================================
OPEN_POS = {}        # { symbol: position_dict }
LAST_CANDLE_TS = {}  # { symbol: last_closed_candle_ts }
ALERT_STATE = {}     # { "symbol|EVENT": last_epoch }

def now_epoch() -> int:
    return int(time.time())

def can_alert(key: str) -> bool:
    last = ALERT_STATE.get(key, 0)
    return (now_epoch() - last) >= COOLDOWN_SEC

def mark_alert(key: str):
    ALERT_STATE[key] = now_epoch()


# =========================================================
# 4) HELPERS
# =========================================================
def capital_per_trade(trading_capital: float) -> float:
    cap = max(BASE_TRADE_CAPITAL, trading_capital * TRADE_PCT)
    cap = min(cap, trading_capital * TRADE_CAP_PCT)
    return round(cap, 2)


# ---- OKX symbol normalization ----
MARKETS = None

def resolve_symbol_for_okx(user_symbol: str) -> str:
    """
    OKX บาง market ใน ccxt อาจเป็น 'BTC/USDT' หรือ 'BTC/USDT:USDT'
    ฟังก์ชันนี้จะพยายาม map ให้เจอตลาดที่มีจริง
    """
    if MARKETS is None:
        return user_symbol

    # exact match
    if user_symbol in MARKETS:
        return user_symbol

    # try with :USDT suffix (common in some derivatives, but sometimes appears)
    alt = f"{user_symbol}:USDT"
    if alt in MARKETS:
        return alt

    # try uppercase normalization
    up = user_symbol.upper()
    if up in MARKETS:
        return up
    alt2 = f"{up}:USDT"
    if alt2 in MARKETS:
        return alt2

    # fallback to original (will error -> handled)
    return user_symbol


def get_data(symbol: str, limit: int = 350):
    sym = resolve_symbol_for_okx(symbol)
    try:
        bars = ccxt_call_with_retry(
            exchange.fetch_ohlcv,
            sym,
            timeframe=TIMEFRAME,
            limit=limit,
            retries=5,
            base_sleep=2.0
        )
        df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        logger.error(f"GET_DATA FAIL | {symbol} ({sym}) | {type(e).__name__}: {e}")

        key = f"{symbol}|DATA_FAIL"
        if can_alert(key):
            send_google_chat(f"⚠️ Data fetch failed for *{symbol}* (OKX): `{type(e).__name__}` (retry next scan)")
            mark_alert(key)
        return None


def add_indicators(df: pd.DataFrame):
    df["ema20"]  = ta.ema(df["close"], length=20)
    df["ema50"]  = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)
    df["rsi"]    = ta.rsi(df["close"], length=14)
    df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atrp"]   = (df["atr"] / df["close"]) * 100.0


def donchian_levels(df: pd.DataFrame, window: int = 20):
    upper = df["high"].rolling(window=window).max()
    lower = df["low"].rolling(window=window).min()
    return lower, upper


# =========================================================
# 5) ENTRY LOGIC (Closed candles only)
# =========================================================
def check_entry(symbol: str, df: pd.DataFrame):
    if df is None or len(df) < 250:
        return None

    add_indicators(df)

    last = df.iloc[-2]  # last closed
    prev = df.iloc[-3]  # previous closed

    entry = float(last["close"])
    atr   = float(last["atr"])
    atrp  = float(last["atrp"])
    rsi   = float(last["rsi"])

    # ATR% filter
    if not (MIN_ATR_PCT <= atrp <= MAX_ATR_PCT):
        return None

    # Donchian (use -3 to avoid repaint)
    dc_low, dc_high = donchian_levels(df, window=20)
    support    = float(dc_low.iloc[-3])
    resistance = float(dc_high.iloc[-3])

    ema20  = float(last["ema20"])
    ema50  = float(last["ema50"])
    ema200 = float(last["ema200"])

    # Volume confirm for breakout
    vol_sma20 = df["volume"].rolling(20).mean().iloc[-2]
    vol_confirm = True
    if pd.notna(vol_sma20):
        vol_confirm = float(last["volume"]) > float(vol_sma20) * 1.2

    # Trend regimes
    uptrend   = (entry > ema200) and (ema20 > ema50)
    downtrend = (entry < ema200) and (ema20 < ema50)

    # Setup A: Pullback EMA20 reclaim
    if ALLOW_LONG and uptrend:
        prev_close = float(prev["close"])
        prev_ema20 = float(prev["ema20"])
        pullback_long = (prev_close < prev_ema20) and (entry > ema20) and (LONG_RSI_MIN <= rsi <= LONG_RSI_MAX)
        if pullback_long:
            return {
                "side": "LONG",
                "setup": "PULLBACK_EMA20",
                "entry": entry,
                "atr": atr,
                "atrp": atrp,
                "rsi": rsi,
                "support": support,
                "resistance": resistance,
            }

    if ALLOW_SHORT and downtrend:
        prev_close = float(prev["close"])
        prev_ema20 = float(prev["ema20"])
        pullback_short = (prev_close > prev_ema20) and (entry < ema20) and (SHORT_RSI_MIN <= rsi <= SHORT_RSI_MAX)
        if pullback_short:
            return {
                "side": "SHORT",
                "setup": "PULLBACK_EMA20",
                "entry": entry,
                "atr": atr,
                "atrp": atrp,
                "rsi": rsi,
                "support": support,
                "resistance": resistance,
            }

    # Setup B: Breakout Donchian20
    if vol_confirm:
        if ALLOW_LONG and uptrend:
            prev_close = float(prev["close"])
            breakout_up = (prev_close <= resistance) and (entry > resistance) and (LONG_RSI_MIN <= rsi <= LONG_RSI_MAX)
            if breakout_up:
                return {
                    "side": "LONG",
                    "setup": "BREAKOUT_DONCHIAN20",
                    "entry": entry,
                    "atr": atr,
                    "atrp": atrp,
                    "rsi": rsi,
                    "support": support,
                    "resistance": resistance,
                }

        if ALLOW_SHORT and downtrend:
            prev_close = float(prev["close"])
            breakout_down = (prev_close >= support) and (entry < support) and (SHORT_RSI_MIN <= rsi <= SHORT_RSI_MAX)
            if breakout_down:
                return {
                    "side": "SHORT",
                    "setup": "BREAKOUT_DONCHIAN20",
                    "entry": entry,
                    "atr": atr,
                    "atrp": atrp,
                    "rsi": rsi,
                    "support": support,
                    "resistance": resistance,
                }

    return None


# =========================================================
# 6) POSITION MANAGEMENT (ENTRY / TP1 / TP2 / SL)
# =========================================================
def open_trade(symbol: str, sig: dict, candle_ts):
    cap = capital_per_trade(TRADING_CAPITAL)
    entry = sig["entry"]
    atr = sig["atr"]
    side = sig["side"]

    qty = cap / entry

    if side == "LONG":
        sl  = entry - SL_ATR * atr
        tp1 = entry + TP1_ATR * atr
        tp2 = entry + TP2_ATR * atr
    else:
        sl  = entry + SL_ATR * atr
        tp1 = entry - TP1_ATR * atr
        tp2 = entry - TP2_ATR * atr

    OPEN_POS[symbol] = {
        "symbol": symbol,
        "status": "OPEN",
        "side": side,
        "setup": sig["setup"],
        "opened_ts": str(candle_ts),
        "entry": entry,
        "qty": qty,
        "cap_used": cap,
        "atr": atr,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp1_done": False,
        "qty_remain": qty,
        "qty_tp1": qty * 0.5,
    }

    msg = (
        f"🚀 *{side} ENTRY* (OKX | {TIMEFRAME} | Closed)\n"
        f"Symbol: *{symbol}*\n"
        f"Setup: *{sig['setup']}*\n"
        f"Entry: `{entry:.4f}`\n"
        f"SL: `{sl:.4f}` | TP1: `{tp1:.4f}` | TP2: `{tp2:.4f}`\n"
        f"Trading Capital: `${TRADING_CAPITAL:.2f}` | Used: `${cap:.2f}` | Qty: `{qty:.6f}`\n"
        f"RSI: `{sig['rsi']:.2f}` | ATR%: `{sig['atrp']:.2f}`\n"
        f"S/R: `{sig['support']:.4f}` / `{sig['resistance']:.4f}`\n"
    )
    send_google_chat(msg)

    logger.info(
        f"ENTRY | {symbol} | {side} | setup={sig['setup']} | entry={entry:.4f} | sl={sl:.4f} | "
        f"tp1={tp1:.4f} | tp2={tp2:.4f} | used={cap:.2f} | qty={qty:.6f} | rsi={sig['rsi']:.2f} | atrp={sig['atrp']:.2f}"
    )


def close_trade(symbol: str, reason: str, exit_price: float):
    pos = OPEN_POS.get(symbol)
    if not pos:
        return

    side = pos["side"]
    entry = float(pos["entry"])
    qty_remain = float(pos["qty_remain"])

    pnl = (exit_price - entry) * qty_remain if side == "LONG" else (entry - exit_price) * qty_remain

    emoji = "🔵" if pnl >= 0 else "🔴"
    msg = (
        f"{emoji} *EXIT* ({reason})\n"
        f"Symbol: *{symbol}* | Side: *{side}*\n"
        f"Exit: `{exit_price:.4f}`\n"
        f"PnL (remain): `${pnl:.2f}`\n"
        f"Setup: {pos.get('setup','-')}\n"
    )
    send_google_chat(msg)
    logger.info(f"EXIT | {symbol} | {side} | reason={reason} | exit={exit_price:.4f} | pnl_remain={pnl:.2f}")

    pos["status"] = "CLOSED"
    del OPEN_POS[symbol]


def manage_trade(symbol: str, candle: pd.Series):
    pos = OPEN_POS.get(symbol)
    if not pos or pos.get("status") != "OPEN":
        return

    side = pos["side"]
    high = float(candle["high"])
    low  = float(candle["low"])

    sl  = float(pos["sl"])
    tp1 = float(pos["tp1"])
    tp2 = float(pos["tp2"])
    entry = float(pos["entry"])

    # STOP LOSS
    if side == "LONG" and low <= sl:
        close_trade(symbol, "STOP LOSS", sl)
        return
    if side == "SHORT" and high >= sl:
        close_trade(symbol, "STOP LOSS", sl)
        return

    # TP2
    if side == "LONG" and high >= tp2:
        close_trade(symbol, "TAKE PROFIT (TP2)", tp2)
        return
    if side == "SHORT" and low <= tp2:
        close_trade(symbol, "TAKE PROFIT (TP2)", tp2)
        return

    # TP1 (partial + move SL to BE)
    if not pos["tp1_done"]:
        tp1_hit = (high >= tp1) if side == "LONG" else (low <= tp1)
        if tp1_hit:
            qty_close = min(float(pos["qty_tp1"]), float(pos["qty_remain"]))

            pnl_part = (tp1 - entry) * qty_close if side == "LONG" else (entry - tp1) * qty_close

            pos["qty_remain"] -= qty_close
            pos["tp1_done"] = True

            msg = (
                f"🟡 *TP1 HIT (Partial)*\n"
                f"Symbol: *{symbol}* | Side: *{side}*\n"
                f"TP1: `{tp1:.4f}` (+{TP1_ATR} ATR)\n"
                f"Closed: `{qty_close:.6f}` | Realized: `${pnl_part:.2f}`\n"
                f"Remaining: `{pos['qty_remain']:.6f}`\n"
            )

            if MOVE_SL_TO_BE_ON_TP1:
                pos["sl"] = entry
                msg += f"Action: Move SL → Break-even (`{entry:.4f}`)\n"

            send_google_chat(msg)

            logger.info(
                f"TP1 | {symbol} | {side} | tp1={tp1:.4f} | qty_closed={qty_close:.6f} | "
                f"realized={pnl_part:.2f} | remain={pos['qty_remain']:.6f} | moveSLBE={MOVE_SL_TO_BE_ON_TP1}"
            )


# =========================================================
# 7) MAIN PROCESS (once per closed candle)
# =========================================================
def process_symbol(symbol: str):
    df = get_data(symbol)
    if df is None or len(df) < 250:
        return

    add_indicators(df)

    candle_ts = df.iloc[-2]["ts"]  # last closed
    if LAST_CANDLE_TS.get(symbol) == candle_ts:
        return
    LAST_CANDLE_TS[symbol] = candle_ts

    # Manage exit first
    if symbol in OPEN_POS:
        manage_trade(symbol, df.iloc[-2])
        return

    # Capacity check
    max_trades = max_open_trades(TRADING_CAPITAL)
    if len(OPEN_POS) >= max_trades:
        return

    sig = check_entry(symbol, df)
    if not sig:
        return

    # Entry cooldown
    key = f"{symbol}|ENTRY"
    if not can_alert(key):
        return

    open_trade(symbol, sig, candle_ts)
    mark_alert(key)


def main():
    logger.info(
        f"SCAN | Open={len(OPEN_POS)}/{max_open_trades(TRADING_CAPITAL)} | TradingCap={TRADING_CAPITAL:.2f} | HODL={HODL_CAPITAL:.2f}"
    )

    for s in CRYPTO_SYMBOLS:
        try:
            process_symbol(s)
        except Exception as e:
            logger.exception(f"PROCESS ERROR | {s} | {e}")


# =========================================================
# 8) RUN
# =========================================================
send_google_chat(
    "🤖 *Bot Online*\n"
    f"OKX {TIMEFRAME} | Long+Short | No leverage\n"
    f"Trading capital=${TRADING_CAPITAL:.0f} (HODL=${HODL_CAPITAL:.0f})\n"
    f"Sizing: max({BASE_TRADE_CAPITAL}, {int(TRADE_PCT*100)}% cap) cap={int(TRADE_CAP_PCT*100)}% | SL={SL_ATR}ATR | TP1={TP1_ATR}ATR | TP2={TP2_ATR}ATR"
)
logger.info(
    f"BOT ONLINE | exchange=OKX | timeframe={TIMEFRAME} | symbols={','.join(CRYPTO_SYMBOLS)} | TradingCap={TRADING_CAPITAL:.2f} | HODL={HODL_CAPITAL:.2f}"
)

# Preload markets once
try:
    MARKETS = ccxt_call_with_retry(exchange.load_markets, retries=5, base_sleep=2.0)
    logger.info(f"MARKETS LOADED OK | markets={len(MARKETS)}")
except Exception as e:
    logger.error(f"MARKETS LOAD FAIL: {e}")
    MARKETS = None

while True:
    try:
        main()
        time.sleep(SCAN_SEC)
    except KeyboardInterrupt:
        logger.info("BOT STOPPED (KeyboardInterrupt)")
        break
    except Exception as e:
        logger.exception(f"LOOP ERROR | {e}")
        time.sleep(60)
