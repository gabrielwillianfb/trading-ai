"""Pure vectorized technical indicator functions for quantitative feature engineering.

All indicators strictly use backward-looking rolling windows to prevent look-ahead bias.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_returns(close: pd.Series, periods: Optional[List[int]] = None) -> pd.DataFrame:
    """Compute simple percentage price returns over multiple backward horizons.

    Formula: (Close_t - Close_{t-k}) / Close_{t-k}

    Args:
        close: Price series.
        periods: List of lag periods (default: [1, 5, 15, 30]).

    Returns:
        DataFrame with columns 'return_{k}'.
    """
    if periods is None:
        periods = [1, 5, 15, 30]

    df = pd.DataFrame(index=close.index)
    for p in periods:
        df[f"return_{p}"] = close.pct_change(periods=p)
    return df


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Compute Simple Moving Average over backward rolling window.

    Args:
        series: Input numerical series.
        window: Number of periods.

    Returns:
        pd.Series of SMA values.
    """
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute Exponential Moving Average.

    Args:
        series: Input numerical series.
        span: Decay span for EMA (e.g. 20, 50, 200).

    Returns:
        pd.Series of EMA values.
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing method.

    Args:
        close: Price series.
        period: RSI period (default: 14).

    Returns:
        pd.Series with values bounded between 0.0 and 100.0.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's exponential smoothing: alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # When loss is 0 and gain > 0, RSI = 100; when gain is 0 and loss > 0, RSI = 0
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)

    return rsi.rename(f"rsi_{period}")


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute Moving Average Convergence Divergence (MACD).

    Args:
        close: Price series.
        fast: Fast EMA span (default: 12).
        slow: Slow EMA span (default: 26).
        signal: Signal line EMA span (default: 9).

    Returns:
        DataFrame with columns 'macd', 'macd_signal', 'macd_hist'.
    """
    fast_ema = compute_ema(close, span=fast)
    slow_ema = compute_ema(close, span=slow)
    macd_line = fast_ema - slow_ema
    signal_line = compute_ema(macd_line, span=signal)
    macd_hist = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_hist,
        },
        index=close.index,
    )


def compute_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands (Upper, Middle, Lower, Width, %B).

    Args:
        close: Price series.
        window: SMA window period (default: 20).
        num_std: Standard deviation multiplier (default: 2.0).

    Returns:
        DataFrame with columns 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent'.
    """
    middle = compute_sma(close, window=window)
    std = close.rolling(window=window, min_periods=window).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    # Bandwidth = (Upper - Lower) / Middle
    bb_width = (upper - lower) / middle.replace(0.0, np.nan)

    # %B = (Close - Lower) / (Upper - Lower)
    band_range = (upper - lower).replace(0.0, np.nan)
    bb_percent = (close - lower) / band_range

    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "bb_width": bb_width,
            "bb_percent": bb_percent,
        },
        index=close.index,
    )


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range (ATR).

    True Range = max(High - Low, |High - Close_prev|, |Low - Close_prev|)

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Smoothing period (default: 14).

    Returns:
        pd.Series of ATR values.
    """
    close_prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr.rename(f"atr_{period}")


def compute_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling historical price volatility based on log returns.

    Args:
        close: Price series.
        window: Rolling window period (default: 20).

    Returns:
        pd.Series of rolling return standard deviations.
    """
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(window=window, min_periods=window).std()
    return vol.rename(f"volatility_{window}")


def compute_volume_features(volume: pd.Series, window: int = 20) -> pd.DataFrame:
    """Compute volume moving average and relative volume ratio.

    Volume Ratio = Volume_t / VolumeSMA20_t

    Args:
        volume: Volume series.
        window: SMA window period (default: 20).

    Returns:
        DataFrame with columns 'volume_sma_{window}', 'volume_ratio'.
    """
    vol_sma = compute_sma(volume, window=window)
    vol_ratio = volume / vol_sma.replace(0.0, np.nan)

    return pd.DataFrame(
        {
            f"volume_sma_{window}": vol_sma,
            "volume_ratio": vol_ratio,
        },
        index=volume.index,
    )


def compute_ema_distances(
    close: pd.Series,
    ema_dict: Dict[str, pd.Series],
) -> pd.DataFrame:
    """Compute relative percentage distance from Close price to each EMA.

    Formula: (Close - EMA) / EMA

    Args:
        close: Price series.
        ema_dict: Mapping of name to EMA series (e.g. {'ema_20': ema_20_series}).

    Returns:
        DataFrame with distance columns 'dist_{name}'.
    """
    df = pd.DataFrame(index=close.index)
    for name, ema in ema_dict.items():
        df[f"dist_{name}"] = (close - ema) / ema.replace(0.0, np.nan)
    return df


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Compute Stochastic Oscillator (%K and %D).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        k_period: Lookback window for %K (default: 14).
        d_period: Moving average window for %D (default: 3).

    Returns:
        DataFrame with columns 'stoch_k', 'stoch_d'.
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    hl_range = (highest_high - lowest_low).replace(0.0, np.nan)

    stoch_k = ((close - lowest_low) / hl_range) * 100.0
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d}, index=close.index)


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Compute Commodity Channel Index (CCI).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Rolling window period (default: 20).

    Returns:
        pd.Series of CCI values.
    """
    tp = (high + low + close) / 3.0
    tp_sma = tp.rolling(window=period, min_periods=period).mean()
    mean_dev = (tp - tp_sma).abs().rolling(window=period, min_periods=1).mean()
    cci = (tp - tp_sma) / (0.015 * mean_dev.replace(0.0, np.nan))
    return cci.rename(f"cci_{period}")


def compute_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Williams %R momentum oscillator.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Lookback window (default: 14).

    Returns:
        pd.Series of Williams %R values bounded between -100 and 0.
    """
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    hl_range = (highest_high - lowest_low).replace(0.0, np.nan)

    williams_r = ((highest_high - close) / hl_range) * -100.0
    return williams_r.rename(f"williams_r_{period}")


def compute_adx_dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Compute Average Directional Index (ADX) and Directional Movement Indicators (+DI, -DI).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Smoothing period (default: 14).

    Returns:
        DataFrame with columns 'adx_{period}', 'dmi_plus_{period}', 'dmi_minus_{period}'.
    """
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    # True Range calculation
    close_prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing
    tr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=close.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=close.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()

    plus_di = (plus_dm_smooth / tr_smooth.replace(0.0, np.nan)) * 100.0
    minus_di = (minus_dm_smooth / tr_smooth.replace(0.0, np.nan)) * 100.0

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / di_sum) * 100.0
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {
            f"adx_{period}": adx,
            f"dmi_plus_{period}": plus_di,
            f"dmi_minus_{period}": minus_di,
        },
        index=close.index,
    )


def compute_donchian_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """Compute Donchian Channel metrics (Width and Relative Position).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        window: Lookback period (default: 20).

    Returns:
        DataFrame with columns 'donchian_width_{window}', 'donchian_pos_{window}'.
    """
    upper = high.rolling(window=window, min_periods=window).max()
    lower = low.rolling(window=window, min_periods=window).min()
    channel_range = (upper - lower).replace(0.0, np.nan)

    width = channel_range / close.replace(0.0, np.nan)
    pos = (close - lower) / channel_range

    return pd.DataFrame(
        {
            f"donchian_width_{window}": width,
            f"donchian_pos_{window}": pos,
        },
        index=close.index,
    )


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Money Flow Index (MFI).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.
        period: Lookback period (default: 14).

    Returns:
        pd.Series of MFI values bounded between 0 and 100.
    """
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    tp_diff = tp.diff()

    pos_flow = pd.Series(np.where(tp_diff > 0, rmf, 0.0), index=close.index)
    neg_flow = pd.Series(np.where(tp_diff < 0, rmf, 0.0), index=close.index)

    pos_sum = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(window=period, min_periods=period).sum()

    mr = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + mr))
    mfi = mfi.where(neg_sum != 0.0, 100.0)
    mfi = mfi.where(pos_sum != 0.0, 0.0)

    return mfi.rename(f"mfi_{period}")


def compute_cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Compute Chaikin Money Flow (CMF).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.
        period: Lookback period (default: 20).

    Returns:
        pd.Series of CMF values.
    """
    hl_range = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * volume

    vol_sum = volume.rolling(window=period, min_periods=period).sum()
    cmf = mfv.rolling(window=period, min_periods=period).sum() / vol_sum.replace(0.0, np.nan)
    return cmf.rename(f"cmf_{period}")


def compute_obv_slope(
    close: pd.Series,
    volume: pd.Series,
    window: int = 10,
) -> pd.Series:
    """Compute On-Balance Volume normalized momentum slope.

    Args:
        close: Price series.
        volume: Volume series.
        window: Slope momentum window (default: 10).

    Returns:
        pd.Series of normalized OBV slope.
    """
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    vol_mean = volume.rolling(window=window, min_periods=window).mean().replace(0.0, np.nan)
    obv_slope = (obv - obv.shift(window)) / (vol_mean * window)
    return obv_slope.rename(f"obv_slope_{window}")


def compute_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute Parkinson Extreme Value Volatility.

    Args:
        high: High price series.
        low: Low price series.
        window: Rolling window period (default: 20).

    Returns:
        pd.Series of Parkinson Volatility values.
    """
    hl_ratio = (high / low.replace(0.0, np.nan)).replace(0.0, np.nan)
    log_hl = np.log(hl_ratio)
    factor = 1.0 / (4.0 * np.log(2.0))
    p_vol = np.sqrt(factor * (log_hl ** 2).rolling(window=window, min_periods=window).mean())
    return pd.Series(p_vol, index=high.index, name=f"parkinson_vol_{window}")


def compute_linear_reg_slope(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute rolling normalized Linear Regression slope of price action.

    Args:
        close: Price series.
        window: Lookback period (default: 14).

    Returns:
        pd.Series of normalized slope (% per bar).
    """
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_dev = x - x_mean
    x_var = (x_dev ** 2).sum()

    def calc_slope(chunk):
        if len(chunk) < window or np.isnan(chunk).any():
            return np.nan
        y_mean = np.mean(chunk)
        slope = np.sum((chunk - y_mean) * x_dev) / x_var
        last_val = chunk[-1]
        return (slope / last_val) * 100.0 if last_val != 0 else 0.0

    slope_series = close.rolling(window=window, min_periods=window).apply(calc_slope, raw=True)
    return slope_series.rename(f"linear_slope_{window}")


def compute_chop_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Compute Choppiness Index (CHOP) to identify trending vs choppy market regimes.

    Values > 61.8 indicate market consolidation/choppiness.
    Values < 38.2 indicate a strong directional trend.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        window: Lookback period (default: 14).

    Returns:
        pd.Series of Choppiness Index values (0.0 to 100.0).
    """
    close_prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_sum = tr.rolling(window=window, min_periods=window).sum()
    highest_high = high.rolling(window=window, min_periods=window).max()
    lowest_low = low.rolling(window=window, min_periods=window).min()
    hl_range = (highest_high - lowest_low).replace(0.0, np.nan)

    ratio = (tr_sum / hl_range).clip(lower=1e-6)
    chop = 100.0 * (np.log10(ratio) / np.log10(window))
    chop = chop.clip(0.0, 100.0)
    return chop.rename(f"chop_index_{window}")
