"""
Stylized facts of financial returns - Stage 1 time series basics.

Runs five checks on S&P 500 (SPY) daily data and prints the results:
  1. Returns      : simple vs log returns
  2. Stationarity : ADF and KPSS tests on prices vs returns
  3. Autocorrelation : Ljung-Box test on returns vs squared returns
  4. Fat tails    : skewness, kurtosis, Jarque-Bera, extreme-day counts
  5. Decomposition: trend / seasonal / residual (CO2 data, which has seasonality)

Usage:
    python stylized_facts.py
    python stylized_facts.py --ticker AAPL --start 2010-01-01 --end 2025-12-31
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.datasets import co2
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

# statsmodels warns about future API changes and about KPSS p-values outside
# its lookup table; neither changes the results, so keep the output clean.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*outside of the range of p-values.*")


def load_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"].squeeze().dropna()
    prices.name = ticker
    return prices


# ---------------------------------------------------------------- 1. Returns
def show_returns(prices):
    simple = prices.pct_change().dropna()
    log = np.log(prices / prices.shift(1)).dropna()

    print("=" * 60)
    print("1. RETURNS")
    print("=" * 60)
    print(f"First 3 days:")
    print(pd.DataFrame({"price": prices.iloc[1:4], "simple": simple.iloc[:3],
                        "log": log.iloc[:3]}).round(5))

    # Log returns add up over time; simple returns do not.
    total_from_prices = prices.iloc[-1] / prices.iloc[0] - 1
    total_from_log = np.exp(log.sum()) - 1
    wrong_from_simple = simple.sum()
    print(f"\nTotal return from prices      : {total_from_prices:8.2%}")
    print(f"exp(sum of log returns) - 1   : {total_from_log:8.2%}  <- same")
    print(f"sum of simple returns (wrong) : {wrong_from_simple:8.2%}  <- differs")
    return log


# ----------------------------------------------------------- 2. Stationarity
def stationarity_tests(series, label):
    adf_p = adfuller(series, autolag="AIC")[1]
    kpss_p = kpss(series, regression="c", nlags="auto")[1]
    # ADF:  H0 = non-stationary -> small p means stationary
    # KPSS: H0 = stationary     -> small p means non-stationary
    adf_says = "stationary" if adf_p < 0.05 else "non-stationary"
    kpss_says = "non-stationary" if kpss_p < 0.05 else "stationary"
    print(f"{label:<8} ADF p={adf_p:6.4f} ({adf_says:<14})  "
          f"KPSS p={kpss_p:6.4f} ({kpss_says})")


def show_stationarity(prices, log_ret):
    print("\n" + "=" * 60)
    print("2. STATIONARITY  (ADF: H0 non-stationary / KPSS: H0 stationary)")
    print("=" * 60)
    stationarity_tests(prices, "Price")
    stationarity_tests(log_ret, "Return")
    print("(KPSS p-values are capped at 0.01 and 0.10 by statsmodels)")


# -------------------------------------------------------- 3. Autocorrelation
def show_autocorrelation(log_ret):
    print("\n" + "=" * 60)
    print("3. AUTOCORRELATION  (Ljung-Box, 10 lags, H0 = no autocorrelation)")
    print("=" * 60)
    for label, series in [("Return", log_ret), ("Return^2", log_ret ** 2)]:
        lb = acorr_ljungbox(series, lags=[10], return_df=True)
        acf1 = series.autocorr(lag=1)
        print(f"{label:<9} lag-1 autocorr={acf1:7.4f}   "
              f"Q={lb['lb_stat'].iloc[0]:9.1f}   p={lb['lb_pvalue'].iloc[0]:.4g}")
    print("Squared returns are far more autocorrelated -> volatility clustering")


# -------------------------------------------------------------- 4. Fat tails
def show_fat_tails(log_ret):
    print("\n" + "=" * 60)
    print("4. FAT TAILS")
    print("=" * 60)
    skew = stats.skew(log_ret)
    kurt = stats.kurtosis(log_ret)          # excess kurtosis: normal = 0
    jb_p = stats.jarque_bera(log_ret).pvalue
    print(f"Skewness        : {skew:7.3f}   (normal = 0)")
    print(f"Excess kurtosis : {kurt:7.3f}   (normal = 0)")
    print(f"Jarque-Bera p   : {jb_p:.4g}   (H0 = normal distribution)")

    z = (log_ret - log_ret.mean()) / log_ret.std()
    n = len(z)
    print(f"\n{'Move':<8}{'Actual days':>12}{'Normal expects':>16}")
    for k in [3, 4, 5]:
        actual = int((z.abs() > k).sum())
        expected = n * 2 * stats.norm.sf(k)
        print(f">{k} sigma{actual:12d}{expected:16.2f}")


# ---------------------------------------------------------- 5. Decomposition
def show_decomposition():
    print("\n" + "=" * 60)
    print("5. DECOMPOSITION  (monthly CO2, Mauna Loa - built into statsmodels)")
    print("=" * 60)
    series = co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
    result = seasonal_decompose(series, model="additive", period=12)
    seasonal = result.seasonal.iloc[:12]
    print("Seasonal effect by month (ppm):")
    for month, value in zip(seasonal.index.strftime("%b"), seasonal):
        print(f"  {month}: {value:+.2f}")
    print("Stock returns have almost no seasonal pattern, so CO2 is used here.")
    return result


# ------------------------------------------------------------------- Chart
def make_chart(prices, log_ret, decomp, ticker, path="stylized_facts.png"):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))

    axes[0, 0].plot(prices, lw=0.8)
    axes[0, 0].set_title(f"{ticker} price - non-stationary (trend)")

    axes[0, 1].plot(log_ret, lw=0.5)
    axes[0, 1].set_title("Log returns - stationary, but volatility clusters")

    plot_acf(log_ret, lags=30, ax=axes[1, 0], zero=False)
    axes[1, 0].set_title("ACF of returns - little autocorrelation")
    axes[1, 0].set_ylim(-0.2, 0.4)

    plot_acf(log_ret ** 2, lags=30, ax=axes[1, 1], zero=False)
    axes[1, 1].set_title("ACF of squared returns - strong autocorrelation")
    axes[1, 1].set_ylim(-0.2, 0.6)

    x = np.linspace(log_ret.min(), log_ret.max(), 400)
    axes[2, 0].hist(log_ret, bins=150, density=True, alpha=0.6, label="Actual")
    axes[2, 0].plot(x, stats.norm.pdf(x, log_ret.mean(), log_ret.std()),
                    "r", lw=1.5, label="Normal")
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_ylim(bottom=1e-2)
    axes[2, 0].set_title("Log-return distribution (log y-axis) - fat tails")
    axes[2, 0].legend()

    axes[2, 1].plot(decomp.observed, lw=0.8, label="Observed")
    axes[2, 1].plot(decomp.trend, lw=1.5, label="Trend")
    axes[2, 1].set_title("CO2 decomposition - trend + seasonality")
    axes[2, 1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"\nChart saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Stylized facts of returns")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    prices = load_prices(args.ticker, args.start, args.end)
    print(f"{args.ticker}: {len(prices)} days, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}\n")

    log_ret = show_returns(prices)
    show_stationarity(prices, log_ret)
    show_autocorrelation(log_ret)
    show_fat_tails(log_ret)
    decomp = show_decomposition()
    make_chart(prices, log_ret, decomp, args.ticker)


if __name__ == "__main__":
    main()
