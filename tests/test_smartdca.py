import numpy as np

from kiwano_portfolio.strategy.smartdca_experiments import simulate_smartdca


def test_standard_dca_matches_closed_form():
    prices = np.array([10.0, 20.0, 25.0])
    result = simulate_smartdca(prices, base_cost=100.0, strategy="dca")
    expected_units = 100 / 10 + 100 / 20 + 100 / 25
    assert np.isclose(result["total_units"], expected_units)
    assert np.isclose(result["total_spent"], 300.0)


def test_rho_zero_equivalent_to_dca():
    prices = np.array([50.0, 40.0, 30.0, 20.0])
    dca = simulate_smartdca(prices, base_cost=20.0, strategy="dca")
    rho0 = simulate_smartdca(prices, base_cost=20.0, strategy="rho", rho=0.0)
    assert np.isclose(dca["total_spent"], rho0["total_spent"])
    assert np.isclose(dca["total_units"], rho0["total_units"])


def test_bounded_out_caps_spend_above_base_cost():
    prices = np.array([100.0, 20.0, 10.0])
    result = simulate_smartdca(prices, base_cost=10.0, strategy="bounded_out", rho=2.0, bound="tanh")
    assert result["total_spent"] <= 30.0


def test_sigplus_changes_spend_profile_vs_dca_after_lookback():
    prices = np.array([100.0, 90.0, 80.0, 120.0, 70.0, 60.0])
    dca = simulate_smartdca(prices, base_cost=10.0, strategy="dca")
    sigplus = simulate_smartdca(
        prices,
        base_cost=10.0,
        strategy="sigplus",
        rho=1.0,
        sigplus_lookback=2,
    )
    assert not np.isclose(dca["total_spent"], sigplus["total_spent"])
