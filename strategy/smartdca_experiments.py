import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    strategy: str
    rho: float
    bound: str
    total_spent: float
    total_units: float
    final_price: float
    final_value: float
    roi: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sin_one(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, 0.0, 1.0)
    return np.sin(0.5 * np.pi * clipped)


def _safe_prices(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    if np.any(prices <= 0):
        raise ValueError("All prices must be strictly positive.")
    return prices


def _compute_sigplus_params(prices: np.ndarray, lookback: int) -> List[Optional[tuple]]:
    inv_prices = 1.0 / prices
    params: List[Optional[tuple]] = []
    for i in range(len(prices)):
        if i < lookback:
            params.append(None)
            continue
        history = inv_prices[i - lookback:i]
        y_max = np.max(history)
        y_min = np.min(history)
        x0 = 0.5 * (y_max + y_min)
        lam = max((y_max - y_min) / 8.0, 1e-12)
        params.append((x0, lam))
    return params


def simulate_smartdca(
    prices: Iterable[float],
    base_cost: float,
    strategy: str = "dca",
    rho: float = 1.0,
    bound: str = "tanh",
    sigplus_lookback: int = 365,
) -> Dict[str, float]:
    """Simulate DCA/SmartDCA using paper formulas.

    strategy:
      - dca
      - rho (unbounded)
      - bounded_out
      - sigplus
    """
    prices_arr = _safe_prices(np.asarray(list(prices), dtype=float))
    reference_price = prices_arr[0]

    if strategy == "dca":
        multipliers = np.ones_like(prices_arr)
    elif strategy == "rho":
        ratio = reference_price / prices_arr
        multipliers = np.power(ratio, rho)
    elif strategy == "bounded_out":
        ratio = reference_price / prices_arr
        if bound == "tanh":
            bounded = np.tanh(ratio)
        elif bound == "sigmoid":
            bounded = _sigmoid(ratio)
        elif bound == "sin-1":
            bounded = _sin_one(ratio)
        else:
            raise ValueError(f"Unknown bound function: {bound}")
        multipliers = np.power(bounded, rho)
    elif strategy == "sigplus":
        params = _compute_sigplus_params(prices_arr, sigplus_lookback)
        inv_prices = 1.0 / prices_arr
        multipliers = np.zeros_like(prices_arr)
        for i, inv_price in enumerate(inv_prices):
            if params[i] is None:
                multipliers[i] = 1.0
            else:
                x0, lam = params[i]
                value = 1.0 / (1.0 + math.exp(-((inv_price - x0) / lam)))
                multipliers[i] = value ** rho
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    spends = base_cost * multipliers
    quantities = spends / prices_arr

    total_spent = float(np.sum(spends))
    total_units = float(np.sum(quantities))
    final_price = float(prices_arr[-1])
    final_value = total_units * final_price
    roi = (final_value / total_spent) - 1.0 if total_spent > 0 else 0.0

    return {
        "total_spent": total_spent,
        "total_units": total_units,
        "final_price": final_price,
        "final_value": final_value,
        "roi": roi,
    }


def fetch_prices_with_kiwano(symbol: str, timeframe: str, lookback: str, end_date: Optional[str]) -> pd.Series:
    try:
        from kiwano_portfolio import Portfolio
    except ModuleNotFoundError:
        from model.portfolio import Portfolio

    portfolio = Portfolio(
        fiat_currency="USD",
        budgets_simulation={"USD": 1_000_000},
        crypto_currencies=[symbol.split("USD")[0]],
        api="yfinance",
        synchronize_wallet=False,
    )
    portfolio.add_strategy(crypto_pair=[symbol], timeframe=timeframe, lookback=lookback)
    parsed_end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
    portfolio.update_data(end_date=parsed_end_date)
    series = portfolio.data[symbol]["Close"].dropna().astype(float)
    if len(series) == 0:
        raise RuntimeError("No price data was fetched. Try another symbol/timeframe/lookback.")
    return series


def run_grid(prices: pd.Series, base_cost: float, rho_values: Iterable[float]) -> pd.DataFrame:
    rows = []
    rows.append({"strategy": "dca", "rho": 0.0, "bound": "none", **simulate_smartdca(prices, base_cost, "dca")})

    for rho in rho_values:
        rows.append({
            "strategy": "rho",
            "rho": rho,
            "bound": "none",
            **simulate_smartdca(prices, base_cost, "rho", rho=rho),
        })
        for bound in ["tanh", "sigmoid", "sin-1"]:
            rows.append({
                "strategy": "bounded_out",
                "rho": rho,
                "bound": bound,
                **simulate_smartdca(prices, base_cost, "bounded_out", rho=rho, bound=bound),
            })
        rows.append({
            "strategy": "sigplus",
            "rho": rho,
            "bound": "sigmoid+",
            **simulate_smartdca(prices, base_cost, "sigplus", rho=rho),
        })

    results = pd.DataFrame(rows)
    return results.sort_values(by=["roi", "final_value"], ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce SmartDCA experiments with kiwano_portfolio data.")
    parser.add_argument("--symbol", default="BTCUSD", help="Yahoo-compatible symbol (e.g. BTCUSD, ETHUSD).")
    parser.add_argument("--timeframe", default="1d", help="Candlestick interval supported by yfinance (e.g. 1d).")
    parser.add_argument("--lookback", default="5y", help="Historical period (e.g. 2y, 5y, 365d).")
    parser.add_argument("--end-date", default=None, help="Optional end date YYYY-MM-DD.")
    parser.add_argument("--base-cost", type=float, default=100.0, help="Base periodic DCA cost c_b.")
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.5, 1.0, 2.0], help="rho values to test.")
    parser.add_argument("--output-csv", default="smartdca_results.csv", help="CSV path for ranked results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = fetch_prices_with_kiwano(args.symbol, args.timeframe, args.lookback, args.end_date)
    results = run_grid(prices, base_cost=args.base_cost, rho_values=args.rhos)
    results.to_csv(args.output_csv, index=False)

    print(f"Fetched {len(prices)} prices for {args.symbol}.")
    print(f"Saved ranked results to {args.output_csv}.")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
