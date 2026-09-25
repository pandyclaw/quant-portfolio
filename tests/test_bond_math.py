"""Tests for models/fixed_income/bond_math.py."""

from __future__ import annotations

from models.fixed_income.bond_math import (
    bond_price,
    compute_analytics,
    convexity,
    dv01,
    macaulay_duration,
    modified_duration,
    price_change_estimate,
)


class TestBondPrice:
    def test_par_bond(self) -> None:
        # A bond priced at par has coupon rate = YTM
        price = bond_price(0.05, 100, 10, 0.05, 2)
        assert abs(price - 100) < 0.01, f"Par bond should price at 100, got {price:.2f}"

    def test_premium_bond(self) -> None:
        # Coupon > YTM → premium
        price = bond_price(0.06, 100, 10, 0.04, 2)
        assert price > 100, "Coupon > YTM should trade at premium"

    def test_discount_bond(self) -> None:
        # Coupon < YTM → discount
        price = bond_price(0.03, 100, 10, 0.05, 2)
        assert price < 100, "Coupon < YTM should trade at discount"

    def test_zero_coupon(self) -> None:
        price = bond_price(0, 100, 10, 0.05, 2)
        expected = 100 / (1.025) ** 20  # ~61.03
        assert abs(price - expected) < 0.01


class TestDuration:
    def test_zero_coupon_duration_equals_maturity(self) -> None:
        # Zero-coupon bond: Macaulay duration = maturity
        dur = macaulay_duration(0, 100, 10, 0.05, 2)
        assert abs(dur - 10.0) < 0.01, f"ZCB duration should equal maturity, got {dur:.2f}"

    def test_coupon_duration_less_than_maturity(self) -> None:
        dur = macaulay_duration(0.05, 100, 10, 0.05, 2)
        assert dur < 10, "Coupon bond duration should be less than maturity"

    def test_modified_less_than_macaulay(self) -> None:
        mac = macaulay_duration(0.05, 100, 10, 0.05, 2)
        mod = modified_duration(0.05, 100, 10, 0.05, 2)
        assert mod < mac, "Modified duration < Macaulay duration"


class TestConvexity:
    def test_convexity_positive(self) -> None:
        conv = convexity(0.05, 100, 10, 0.05, 2)
        assert conv > 0, "Convexity should be positive for standard bonds"

    def test_longer_maturity_more_convex(self) -> None:
        conv_5y = convexity(0.05, 100, 5, 0.05, 2)
        conv_30y = convexity(0.05, 100, 30, 0.05, 2)
        assert conv_30y > conv_5y, "Longer maturity = more convexity"


class TestDV01:
    def test_dv01_positive(self) -> None:
        d = dv01(0.05, 100, 10, 0.05, 2)
        assert d > 0, "DV01 should be positive"

    def test_dv01_approximation(self) -> None:
        # DV01 should approximate actual price change for 1bp move
        p0 = bond_price(0.05, 100, 10, 0.05, 2)
        p1 = bond_price(0.05, 100, 10, 0.0501, 2)
        actual_change = abs(p0 - p1)
        d = dv01(0.05, 100, 10, 0.05, 2)
        assert abs(d - actual_change) < 0.01, f"DV01={d:.4f} vs actual={actual_change:.4f}"


class TestPriceChangeEstimate:
    def test_small_move(self) -> None:
        mod = modified_duration(0.05, 100, 10, 0.05, 2)
        conv = convexity(0.05, 100, 10, 0.05, 2)
        dy = 0.001  # 10bp
        est = price_change_estimate(mod, conv, dy)
        # Should be approximately -D*dy for small moves
        assert est < 0, "Price should fall when yields rise"


class TestComputeAnalytics:
    def test_returns_all_fields(self) -> None:
        result = compute_analytics(0.05, 100, 10, 0.05, 2)
        assert result.price > 0
        assert result.macaulay_duration > 0
        assert result.modified_duration > 0
        assert result.convexity > 0
        assert result.dv01 > 0
