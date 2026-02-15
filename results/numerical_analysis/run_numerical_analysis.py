import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


OUT_DIR = Path("results/numerical_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StrategySpec:
    name: str
    kind: str
    rho: float = 0.0
    xi: int = 1
    lam: float = 0.0
    window: int = 90
    rho_max: float = 2.0
    kappa: float = 1.0


def fetch_prices(symbol: str, start: str, end: str) -> pd.Series:
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            s = df["Close"].iloc[:, 0].dropna().astype(float)
        else:
            s = df.iloc[:, 0].dropna().astype(float)
    else:
        col = "Close" if "Close" in df.columns else df.columns[0]
        s = df[col].dropna().astype(float)
    s.name = symbol
    return s


def moment_weight(r, rho, xi, lam):
    # with f=tanh and normalized by asymptotic supremum (=1 for xi=1 else +inf); we cap by max on observed sample
    base = np.tanh(r) ** rho
    raw = (1 - lam) * base + lam * (r ** (xi - 1)) * base
    return raw


def tail_indicator(r_window: np.ndarray, xi: int = 2, rho0: float = 0.0):
    w = np.tanh(r_window) ** rho0
    num = np.sum((r_window ** xi) * w) / np.sum((r_window ** (xi - 1)) * w)
    den = np.sum((r_window ** 1) * w) / np.sum((r_window ** 0) * w)
    den = max(den, 1e-12)
    return num / (den ** xi)


def simulate(prices: pd.Series, spec: StrategySpec):
    p = prices.to_numpy()
    r = p[0] / p
    cb = 1.0

    if spec.kind == "dca":
        c = np.full_like(r, cb)
    elif spec.kind == "rho_unbounded":
        c = cb * (r ** spec.rho)
    elif spec.kind == "bounded_tanh":
        c = cb * (np.tanh(r) ** spec.rho)
    elif spec.kind == "moment_tilted":
        raw = moment_weight(r, spec.rho, spec.xi, spec.lam)
        norm = max(raw.max(), 1e-12)
        c = cb * raw / norm
    elif spec.kind == "adaptive_rho":
        rho_t = np.zeros_like(r)
        T = np.ones_like(r)
        for i in range(len(r)):
            left = max(0, i - spec.window + 1)
            rw = r[left : i + 1]
            T[i] = tail_indicator(rw, xi=2, rho0=0.0)
            rho_t[i] = np.clip(0.0 + spec.kappa * np.log(max(T[i], 1e-12)), 0.0, spec.rho_max)
        c = cb * (np.tanh(r) ** rho_t)
    else:
        raise ValueError(spec.kind)

    q = c / p
    C = np.cumsum(c)
    Q = np.cumsum(q)
    mu_t = C / Q
    roi_t = (Q * p - C) / C

    metrics = {
        "mu": float(mu_t[-1]),
        "Q": float(Q[-1]),
        "C": float(C[-1]),
        "ROI": float(roi_t[-1]),
        "max_c": float(c.max()),
        "cv_c": float(np.std(c) / np.mean(c)),
    }
    return metrics, pd.DataFrame({"price": p, "c": c, "C": C, "Q": Q, "mu": mu_t, "roi": roi_t}, index=prices.index)


def choose_best_moment_btc(prices: pd.Series):
    split = int(len(prices) * 0.6)
    train, val = prices.iloc[:split], prices.iloc[split:]
    best = None
    for xi in [2, 3]:
        for lam in [0.25, 0.5, 0.75]:
            spec = StrategySpec(name=f"moment_xi{xi}_lam{lam}", kind="moment_tilted", rho=2, xi=xi, lam=lam)
            _, _ = simulate(train, spec)
            m_val, _ = simulate(val, spec)
            score = m_val["ROI"]
            if best is None or score > best[0]:
                best = (score, spec)
    return best[1]


def choose_best_adaptive(prices: pd.Series):
    split = int(len(prices) * 0.6)
    train, val = prices.iloc[:split], prices.iloc[split:]
    best = None
    for W in [90, 252]:
        for rho_max in [2, 3]:
            for kappa in [0.5, 1.0, 1.5, 2.0, 3.0]:
                spec = StrategySpec(name=f"adaptive_W{W}_rmax{rho_max}_k{kappa}", kind="adaptive_rho", window=W, rho_max=rho_max, kappa=kappa)
                _, _ = simulate(train, spec)
                m_val, _ = simulate(val, spec)
                score = m_val["ROI"]
                if best is None or score > best[0]:
                    best = (score, spec)
    return best[1]


def make_strategy_list(moment_best: StrategySpec, adaptive_best: StrategySpec):
    return [
        StrategySpec("DCA", "dca"),
        StrategySpec("rho=1", "rho_unbounded", rho=1),
        StrategySpec("rho=2", "rho_unbounded", rho=2),
        StrategySpec("rho=3", "rho_unbounded", rho=3),
        StrategySpec("tanh_rho=1", "bounded_tanh", rho=1),
        StrategySpec("tanh_rho=2", "bounded_tanh", rho=2),
        StrategySpec(
            f"moment_tilted(rho=2,xi={moment_best.xi},lam={moment_best.lam})",
            "moment_tilted",
            rho=2,
            xi=moment_best.xi,
            lam=moment_best.lam,
        ),
        StrategySpec(
            f"adaptive(W={adaptive_best.window},rho_max={adaptive_best.rho_max},k={adaptive_best.kappa})",
            "adaptive_rho",
            window=adaptive_best.window,
            rho_max=adaptive_best.rho_max,
            kappa=adaptive_best.kappa,
        ),
    ]


def run_block_roi(prices: pd.Series, blocks, strategies):
    out = []
    for bname, start, end in blocks:
        p = prices[(prices.index >= pd.Timestamp(start)) & (prices.index < pd.Timestamp(end))]
        if len(p) < 30:
            continue
        for s in strategies:
            m, _ = simulate(p, s)
            out.append({"block": bname, "strategy": s.name, "ROI": m["ROI"]})
    return pd.DataFrame(out)


def plot_n1_synthetic(strategies):
    rng = np.random.default_rng(123)
    p = pd.Series(rng.uniform(0.01, 2.0, 1000))
    rho_grid = np.linspace(0, 3, 25)
    mus = []
    for rho in rho_grid:
        m_dca, _ = simulate(p, StrategySpec("DCA", "dca"))
        m_rho, _ = simulate(p, StrategySpec("rho", "rho_unbounded", rho=rho))
        m_bt, _ = simulate(p, StrategySpec("tanh", "bounded_tanh", rho=rho))
        m_mom, _ = simulate(p, StrategySpec("mom", "moment_tilted", rho=max(rho, 0.1), xi=2, lam=0.5))
        m_ad, _ = simulate(p, StrategySpec("ad", "adaptive_rho", window=90, rho_max=3, kappa=1.5))
        mus.append([rho, m_dca["mu"], m_rho["mu"], m_bt["mu"], m_mom["mu"], m_ad["mu"]])
    df = pd.DataFrame(mus, columns=["rho", "DCA", "Unbounded", "Bounded", "Moment", "Adaptive"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for c in ["DCA", "Unbounded", "Bounded", "Moment", "Adaptive"]:
        axes[0].plot(df["rho"], df[c], label=c)
    axes[0].set_title("Figure N1a: synthetic $\\mu$ vs $\\rho$")
    axes[0].set_xlabel("rho")
    axes[0].set_ylabel("Average purchase price mu")
    axes[0].legend(fontsize=8)

    _, traj = simulate(p, StrategySpec("moment", "moment_tilted", rho=2, xi=2, lam=0.5))
    axes[1].plot(traj["c"].values[:200])
    axes[1].set_title("Figure N1b: example contributions (first 200 steps)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("c_i")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_n1_synthetic.png", dpi=160)
    plt.close(fig)


def plot_trajectories(asset_name, prices, strategies):
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in strategies:
        _, tr = simulate(prices, s)
        ax.plot(tr.index, tr["roi"], label=s.name)
    ax.set_title(f"Figure N2: Cumulative ROI trajectories ({asset_name})")
    ax.set_ylabel("ROI")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"figure_n2_{asset_name.lower().replace('&','and').replace(' ','_')}.png", dpi=160)
    plt.close(fig)


def plot_subperiod_bars(df_blocks, asset_name):
    pivot = df_blocks.pivot(index="block", columns="strategy", values="ROI")
    pivot.plot(kind="bar", figsize=(12, 4))
    plt.title(f"Figure N3: Subperiod ROI ({asset_name})")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"figure_n3_{asset_name.lower().replace('&','and').replace(' ','_')}.png", dpi=160)
    plt.close()


def main():
    sp = fetch_prices("^GSPC", "1973-01-01", "2024-01-01")
    btc = fetch_prices("BTC-USD", "2017-01-01", "2024-01-01")

    best_moment_btc = choose_best_moment_btc(btc)
    best_adapt_btc = choose_best_adaptive(btc)
    strategies = make_strategy_list(best_moment_btc, best_adapt_btc)

    summary_rows = []
    traj_cache = {}
    for asset_name, prices in [("SP500", sp), ("BTC", btc)]:
        for s in strategies:
            m, tr = simulate(prices, s)
            traj_cache[(asset_name, s.name)] = tr
            summary_rows.append({"asset": asset_name, "strategy": s.name, **m})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "headline_summary.csv", index=False)

    # table N1 for BTC
    btc_table = summary[summary["asset"] == "BTC"].copy()
    btc_table.to_csv(OUT_DIR / "table_n1_btc_2017_2023.csv", index=False)

    # blocks
    sp_blocks = []
    for y in range(1973, 2024, 5):
        y2 = min(y + 5, 2024)
        sp_blocks.append((f"{y}-{y2}", f"{y}-01-01", f"{y2}-01-01"))
    btc_blocks = []
    for y in range(2018, 2023):
        btc_blocks.append((f"{y}-{y+1}", f"{y}-01-01", f"{y+1}-01-01"))

    sp_block_df = run_block_roi(sp, sp_blocks, strategies)
    btc_block_df = run_block_roi(btc, btc_blocks, strategies)
    sp_block_df.to_csv(OUT_DIR / "sp500_subperiod_roi.csv", index=False)
    btc_block_df.to_csv(OUT_DIR / "btc_subperiod_roi.csv", index=False)

    # beat DCA fractions
    beat_rows = []
    for asset, df in [("SP500", sp_block_df), ("BTC", btc_block_df)]:
        piv = df.pivot(index="block", columns="strategy", values="ROI")
        for col in piv.columns:
            if col == "DCA":
                continue
            beat_rows.append({"asset": asset, "strategy": col, "beat_dca_fraction": float((piv[col] > piv["DCA"]).mean())})
    pd.DataFrame(beat_rows).to_csv(OUT_DIR / "beat_dca_fraction.csv", index=False)

    # ablations
    ablation_moment = []
    for xi in [2, 3]:
        for lam in [0.25, 0.5, 0.75]:
            s = StrategySpec("ab", "moment_tilted", rho=2, xi=xi, lam=lam)
            m, _ = simulate(btc, s)
            ablation_moment.append({"xi": xi, "lambda": lam, **m})
    pd.DataFrame(ablation_moment).to_csv(OUT_DIR / "ablation_moment_tilt_btc.csv", index=False)

    ablation_adapt = []
    for k in [0.5, 1.0, 1.5, 2.0, 3.0]:
        s = StrategySpec("ab", "adaptive_rho", window=best_adapt_btc.window, rho_max=best_adapt_btc.rho_max, kappa=k)
        m, _ = simulate(btc, s)
        ablation_adapt.append({"window": best_adapt_btc.window, "rho_max": best_adapt_btc.rho_max, "kappa": k, **m})
    pd.DataFrame(ablation_adapt).to_csv(OUT_DIR / "ablation_adaptive_kappa_btc.csv", index=False)

    plot_n1_synthetic(strategies)
    plot_trajectories("SP500", sp, strategies)
    plot_trajectories("BTC", btc, strategies)
    plot_subperiod_bars(sp_block_df, "SP500")
    plot_subperiod_bars(btc_block_df, "BTC")

    metadata = {
        "best_moment_btc": best_moment_btc.__dict__,
        "best_adaptive_btc": best_adapt_btc.__dict__,
        "strategies": [s.__dict__ for s in strategies],
        "n_sp500": len(sp),
        "n_btc": len(btc),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
