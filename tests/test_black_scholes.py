"""Tests for models/options/black_scholes.py."""

from __future__ import annotations

import numpy as np

from models.options.black_scholes import (
    call_price,
    greeks,
    implied_volatility,
    put_call_parity_check,
    put_price,
)


class TestPricing:
    def test_call_atm(self) -> None:
        # ATM call: S=K=100, T=1, r=5%, vol=20%
        price = call_price(100, 100, 1.0, 0.05, 0.20)
        assert 8 < price < 12, f"ATM 1Y call should be ~$10, got {price:.2f}"

    def test_put_atm(self) -> None:
        price = put_price(100, 100, 1.0, 0.05, 0.20)
        assert 4 < price < 8, f"ATM 1Y put should be ~$6, got {price:.2f}"

    def test_call_deep_itm(self) -> None:
        price = call_price(150, 100, 1.0, 0.05, 0.20)
        assert price > 45, "Deep ITM call should be worth at least intrinsic"

    def test_put_deep_itm(self) -> None:
        price = put_price(50, 100, 1.0, 0.05, 0.20)
        assert price > 45, "Deep ITM put should be worth at least intrinsic"

    def test_expired_call(self) -> None:
        assert call_price(110, 100, 0, 0.05, 0.20) == 10.0
        assert call_price(90, 100, 0, 0.05, 0.20) == 0.0

    def test_expired_put(self) -> None:
        assert put_price(90, 100, 0, 0.05, 0.20) == 10.0
        assert put_price(110, 100, 0, 0.05, 0.20) == 0.0

    def test_higher_vol_higher_price(self) -> None:
        low = call_price(100, 100, 1.0, 0.05, 0.10)
        high = call_price(100, 100, 1.0, 0.05, 0.40)
        assert high > low, "Higher vol should mean higher option price"


class TestPutCallParity:
    def test_parity_holds(self) -> None:
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        c = call_price(S, K, T, r, sigma)
        p = put_price(S, K, T, r, sigma)
        deviation = put_call_parity_check(c, p, S, K, T, r)
        assert abs(deviation) < 1e-6, f"Put-call parity violated: {deviation}"


class TestGreeks:
    def test_call_delta_positive(self) -> None:
        g = greeks(100, 100, 1.0, 0.05, 0.20, "call")
        assert 0.4 < g.delta < 0.7, f"ATM call delta should be ~0.5, got {g.delta:.3f}"

    def test_put_delta_negative(self) -> None:
        g = greeks(100, 100, 1.0, 0.05, 0.20, "put")
        assert -0.7 < g.delta < -0.3, f"ATM put delta should be ~-0.5, got {g.delta:.3f}"

    def test_gamma_positive(self) -> None:
        g = greeks(100, 100, 1.0, 0.05, 0.20, "call")
        assert g.gamma > 0, "Gamma is always positive for long options"

    def test_vega_positive(self) -> None:
        g = greeks(100, 100, 1.0, 0.05, 0.20, "call")
        assert g.vega > 0, "Vega is always positive for long options"

    def test_theta_negative_for_long(self) -> None:
        g = greeks(100, 100, 1.0, 0.05, 0.20, "call")
        # Theta is negative because we compute for long position (time decay hurts)
        assert g.theta < 0, "Long option theta should be negative"


class TestImpliedVol:
    def test_roundtrip(self) -> None:
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        price = call_price(S, K, T, r, sigma)
        iv = implied_volatility(price, S, K, T, r, "call")
        assert abs(iv - sigma) < 0.001, f"IV should recover input vol: {iv:.4f} vs {sigma}"

    def test_put_roundtrip(self) -> None:
        S, K, T, r, sigma = 100, 95, 0.5, 0.03, 0.30
        price = put_price(S, K, T, r, sigma)
        iv = implied_volatility(price, S, K, T, r, "put")
        assert abs(iv - sigma) < 0.001

    def test_high_vol(self) -> None:
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.80
        price = call_price(S, K, T, r, sigma)
        iv = implied_volatility(price, S, K, T, r, "call")
        assert abs(iv - sigma) < 0.01

    def test_expired_returns_nan(self) -> None:
        iv = implied_volatility(10, 100, 90, 0, 0.05, "call")
        assert np.isnan(iv)
