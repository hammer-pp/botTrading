# backtest_cdc_actionzone_winrate_pack_5y.py
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import timedelta

# =========================
# CONFIG
# =========================
SYMBOLS = ["AAPL",
"MSFT",
"AMZN",
"GOOGL",
"META",
"NVDA",
"TSLA"]

# Try 1h first; if not enough bars for 5y -> fallback to 1d automatically
PRIMARY_INTERVAL = "1h"
FALLBACK_INTERVAL = "1d"

LOOKBACK_DAYS = 365 * 5    # ✅ 5 years
START_CAPITAL = 1000.0
FEE_RATE = 0.0015           # 0.15% per side

MODE = "ALL_IN"             # CDC style buy/sell all-in
FIXED_BUY_USD = 100.0       # unused when ALL_IN

# CDC ActionZone core
FAST_EMA = 12
SLOW_EMA = 26

# Winrate pack
EMA_TREND_LEN = 200
ADX_LEN = 14
ADX_MIN = 18.0

CONFIRM_BARS = 2
COOLDOWN_AFTER_SELL_BARS = 8
USE_TURN_ONLY = True

# =========================
# HELPERS
# =========================
def annualized_return_pct(pnl_pct: float, days: int) -> float | None:
    if pnl_pct is None or days <= 0:
        return None
    r = pnl_pct / 100.0
    if r <= -1.0:
        return -100.0
    ann = (1.0 + r) ** (365.0 / float(days)) - 1.0
    return ann * 100.0

def fetch_ohlcv(symbol: str, interval: str) -> pd.DataFrame:
    # For long lookbacks, use period with some buffer for warmup
    # yfinance period max depends on interval; we try "max" then slice by LOOKBACK_DAYS
    df = yf.download(symbol, period="max", interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy().reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "datetime" not in df.columns:
        if "date" in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)
        elif "index" in df.columns:
            df.rename(columns={"index": "datetime"}, inplace=True)
        elif "level_0" in df.columns:
            df.rename(columns={"level_0": "datetime"}, inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["datetime", "open", "high", "low", "close"], inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_fast"] = ta.ema(df["close"], length=FAST_EMA)
    df["ema_slow"] = ta.ema(df["close"], length=SLOW_EMA)
    df["ema200"] = ta.ema(df["close"], length=EMA_TREND_LEN)

    adx = ta.adx(df["high"], df["low"], df["close"], length=ADX_LEN)
    adx_col = f"ADX_{ADX_LEN}"
    df["adx"] = adx[adx_col] if adx is not None and hasattr(adx, "columns") and adx_col in adx.columns else pd.NA

    df["cdc_green_raw"] = (df["ema_fast"] > df["ema_slow"]) & (df["close"] > df["ema_fast"])
    df["cdc_red_raw"] = (df["ema_fast"] < df["ema_slow"]) & (df["close"] < df["ema_fast"])

    def state(row):
        if bool(row["cdc_green_raw"]):
            return "GREEN"
        if bool(row["cdc_red_raw"]):
            return "RED"
        return "OTHER"

    df["cdc_state"] = df.apply(state, axis=1)
    return df

def is_confirmed_state(df: pd.DataFrame, i: int, state: str, confirm_bars: int) -> bool:
    if i - (confirm_bars - 1) < 0:
        return False
    for k in range(i - (confirm_bars - 1), i + 1):
        if df["cdc_state"].iloc[k] != state:
            return False
    return True

def was_confirmed_state(df: pd.DataFrame, i: int, state: str, confirm_bars: int) -> bool:
    if i - 1 < 0:
        return False
    return is_confirmed_state(df, i - 1, state, confirm_bars)

def slice_lookback(df: pd.DataFrame, lookback_days: int) -> tuple[pd.DataFrame, int, pd.Timestamp, pd.Timestamp]:
    end_dt = df["datetime"].iloc[-1]
    start_dt = end_dt - timedelta(days=lookback_days)
    idx = df.index[df["datetime"] >= start_dt]
    if len(idx) == 0:
        return df, 0, start_dt, end_dt
    start_i = int(idx[0])
    return df, start_i, start_dt, end_dt

def backtest_symbol(symbol: str) -> dict:
    # Try primary interval
    df = fetch_ohlcv(symbol, PRIMARY_INTERVAL)
    used_interval = PRIMARY_INTERVAL

    if df.empty:
        # fallback
        df = fetch_ohlcv(symbol, FALLBACK_INTERVAL)
        used_interval = FALLBACK_INTERVAL

    # slice to 5y window if possible
    df, start_i, start_dt, end_dt = slice_lookback(df, LOOKBACK_DAYS)

    # Need enough warmup bars for EMA200/ADX
    # If 1h data is too short to cover 5y, fallback to 1d automatically
    min_bars = 260  # warmup
    bars_in_window = len(df) - start_i

    if used_interval == PRIMARY_INTERVAL:
        # if start_i is 0 and dataset doesn't reach 5y (very likely) or too few bars, fallback
        # Heuristic: require at least ~ 252 trading days/year * 5 = 1260 daily bars equivalent
        # For 1h, we'd expect far more than 1260 bars; if dataset is small, it's not full 5y.
        if bars_in_window < 3000:  # conservative threshold for 1h across years
            df = fetch_ohlcv(symbol, FALLBACK_INTERVAL)
            used_interval = FALLBACK_INTERVAL
            if df.empty:
                return {"symbol": symbol, "error": "no data"}
            df, start_i, start_dt, end_dt = slice_lookback(df, LOOKBACK_DAYS)

    if df.empty or len(df) < min_bars or (len(df) - start_i) < 50:
        return {"symbol": symbol, "error": "insufficient data"}

    df = add_indicators(df)

    cash = START_CAPITAL
    shares = 0.0
    invested_gross = 0.0
    invested_buy_fees = 0.0
    entry_time = None
    entry_price_first = None

    cooldown_until = -1
    trades = []

    def do_buy(price: float, t) -> bool:
        nonlocal cash, shares, invested_gross, invested_buy_fees, entry_time, entry_price_first
        if cash <= 0:
            return False
        usd = cash  # ALL_IN

        fee = usd * FEE_RATE
        usd_net = usd - fee
        sh = usd_net / price

        invested_gross += usd
        invested_buy_fees += fee
        shares += sh
        cash -= usd

        if entry_time is None:
            entry_time = t
            entry_price_first = price
        return True

    def do_sell(price: float, t, reason: str) -> None:
        nonlocal cash, shares, invested_gross, invested_buy_fees, entry_time, entry_price_first
        if shares <= 0:
            return

        gross = shares * price
        fee = gross * FEE_RATE
        net = gross - fee

        invested_net = invested_gross - invested_buy_fees
        pnl = net - invested_net

        trades.append({
            "symbol": symbol,
            "interval": used_interval,
            "entry_time": entry_time,
            "exit_time": t,
            "entry_price_first": entry_price_first,
            "exit_price": price,
            "shares": shares,
            "invested_gross_usd": invested_gross,
            "pnl_usd": pnl,
            "reason": reason,
        })

        cash += net
        shares = 0.0
        invested_gross = 0.0
        invested_buy_fees = 0.0
        entry_time = None
        entry_price_first = None

    warmup = max(start_i, 260)
    for i in range(warmup, len(df)):
        t = df["datetime"].iloc[i]
        c = float(df["close"].iloc[i])

        ema200 = df["ema200"].iloc[i]
        adx = df["adx"].iloc[i]

        trend_ok = pd.notna(ema200) and (c > float(ema200))
        adx_ok = pd.notna(adx) and (float(adx) > ADX_MIN)

        green_confirmed = is_confirmed_state(df, i, "GREEN", CONFIRM_BARS)
        red_confirmed = is_confirmed_state(df, i, "RED", CONFIRM_BARS)

        if USE_TURN_ONLY:
            green_turn = green_confirmed and (not was_confirmed_state(df, i, "GREEN", CONFIRM_BARS))
            red_turn = red_confirmed and (not was_confirmed_state(df, i, "RED", CONFIRM_BARS))
        else:
            green_turn = green_confirmed
            red_turn = red_confirmed

        buy_sig = green_turn and trend_ok and adx_ok and (i > cooldown_until)
        sell_sig = red_turn

        if shares <= 0:
            if buy_sig:
                do_buy(c, t)
        else:
            if sell_sig:
                do_sell(c, t, "CDC confirmed RED")
                cooldown_until = i + COOLDOWN_AFTER_SELL_BARS

    # Force exit
    if shares > 0:
        do_sell(float(df["close"].iloc[-1]), df["datetime"].iloc[-1], "FORCE end")

    trades_df = pd.DataFrame(trades)

    end_equity = cash
    pnl = end_equity - START_CAPITAL
    pnl_pct = (pnl / START_CAPITAL) * 100.0
    ann_pct = annualized_return_pct(pnl_pct, LOOKBACK_DAYS)

    stats = {
        "symbol": symbol,
        "interval_used": used_interval,
        "start": str(start_dt),
        "end": str(end_dt),
        "pnl_usd": float(pnl),
        "pnl_pct": float(pnl_pct),
        "annualized_return_pct": float(ann_pct) if ann_pct is not None else None,
        "end_equity": float(end_equity),
        "positions": int(len(trades_df)),
        "win_rate_pct": float((trades_df["pnl_usd"] > 0).mean() * 100.0) if not trades_df.empty else None,
        "exit_breakdown": trades_df["reason"].value_counts().to_dict() if not trades_df.empty else {},
    }
    return {"stats": stats, "trades": trades_df}

def main():
    rows = []
    all_trades = []

    for s in SYMBOLS:
        res = backtest_symbol(s)
        if "error" in res:
            rows.append({"symbol": s, "error": res["error"]})
            continue
        rows.append(res["stats"])
        if not res["trades"].empty:
            all_trades.append(res["trades"])

    summary = pd.DataFrame(rows)

    print("\n===== CDC ACTIONZONE (Winrate Pack) | 5-Year Backtest =====")
    print(f"LOOKBACK_DAYS={LOOKBACK_DAYS} | PRIMARY={PRIMARY_INTERVAL} | FALLBACK={FALLBACK_INTERVAL} | FEE_RATE={FEE_RATE} (0.15%/side)")
    cols = ["symbol","interval_used","pnl_usd","pnl_pct","annualized_return_pct","end_equity","positions","win_rate_pct","exit_breakdown","start","end"]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))

    if "pnl_usd" in summary.columns:
        total_pnl = summary["pnl_usd"].fillna(0).sum()
        total_start = START_CAPITAL * len(SYMBOLS)
        total_end = total_start + total_pnl
        total_pnl_pct = (total_pnl / total_start) * 100.0
        total_ann_pct = annualized_return_pct(total_pnl_pct, LOOKBACK_DAYS)

        print("\n===== TOTAL (1000 USD per symbol) =====")
        print(f"Start Total: {total_start:,.2f} USD")
        print(f"End Total  : {total_end:,.2f} USD")
        print(f"P/L Total  : {total_pnl:,.2f} USD ({total_pnl_pct:.2f}%)")
        if total_ann_pct is not None:
            print(f"Annualized : {total_ann_pct:.2f}% (compound)")

    if all_trades:
        trades_all = pd.concat(all_trades, ignore_index=True)
        print("\n===== LAST 10 TRADES (ALL SYMBOLS) =====")
        print(trades_all.sort_values("exit_time").tail(10).to_string(index=False))

    print("\n===== CURRENT SETTINGS =====")
    print(f"CONFIRM_BARS={CONFIRM_BARS} | USE_TURN_ONLY={USE_TURN_ONLY}")
    print(f"EMA200 filter: close > EMA{EMA_TREND_LEN}")
    print(f"ADX filter: ADX{ADX_LEN} > {ADX_MIN}")
    print(f"COOLDOWN_AFTER_SELL_BARS={COOLDOWN_AFTER_SELL_BARS}")
    print("Note: If 1h data is not available for full 5y via yfinance, script falls back to 1d automatically.")

if __name__ == "__main__":
    # pip install yfinance pandas pandas_ta
    main()
