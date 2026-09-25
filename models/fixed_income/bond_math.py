"""Bond pricing, duration, convexity, and DV01.

Fundamental fixed income analytics. These calculations underpin
all bond portfolio management: duration measures interest rate
sensitivity, convexity measures the curvature of the price/yield
relationship, and DV01 is the dollar value of a 1bp yield move.

Duration + convexity approximation:
    dP/P ≈ -D_mod * dy + 0.5 * C * dy²

where D_mod is modified duration, C is convexity, and dy is the
yield change.

References:
    Fabozzi, F. J. (2016). "Bond Markets, Analysis, and Strategies."
        Pearson.
    Tuckman, B. & Serrat, A. (2011). "Fixed Income Securities." Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BondAnalytics:
    """Complete bond analytics result."""

    price: float
    ytm: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float


def bond_price(
    coupon_rate: float,
    face: float,
    maturity: float,
    ytm: float,
    freq: int = 2,
) -> float:
    """Price a fixed-coupon bond.

    Parameters
    ----------
    coupon_rate
        Annual coupon rate (e.g., 0.05 = 5%).
    face
        Face (par) value.
    maturity
        Years to maturity.
    ytm
        Yield to maturity (annualized).
    freq
        Coupon frequency per year (2 = semi-annual).
    """
    n_periods = int(maturity * freq)
    coupon = face * coupon_rate / freq
    y = ytm / freq

    if y == 0:
        return coupon * n_periods + face

    pv_coupons = coupon * (1 - (1 + y) ** -n_periods) / y
    pv_face = face / (1 + y) ** n_periods

    return float(pv_coupons + pv_face)


def macaulay_duration(
    coupon_rate: float,
    face: float,
    maturity: float,
    ytm: float,
    freq: int = 2,
) -> float:
    """Macaulay duration: weighted average time to cash flows.

    Measured in years. The key interpretation: a bond with Macaulay
    duration of 7 years behaves like a 7-year zero-coupon bond
    in terms of interest rate sensitivity.
    """
    n_periods = int(maturity * freq)
    coupon = face * coupon_rate / freq
    y = ytm / freq
    price = bond_price(coupon_rate, face, maturity, ytm, freq)

    if price <= 0:
        return 0.0

    weighted_sum = 0.0
    for t in range(1, n_periods + 1):
        cf = coupon if t < n_periods else coupon + face
        pv = cf / (1 + y) ** t
        weighted_sum += (t / freq) * pv

    return float(weighted_sum / price)


def modified_duration(
    coupon_rate: float,
    face: float,
    maturity: float,
    ytm: float,
    freq: int = 2,
) -> float:
    """Modified duration = Macaulay duration / (1 + y/freq).

    Measures the percentage price change per 1% yield change:
        dP/P ≈ -D_mod * dy
    """
    mac_dur = macaulay_duration(coupon_rate, face, maturity, ytm, freq)
    return float(mac_dur / (1 + ytm / freq))


def convexity(
    coupon_rate: float,
    face: float,
    maturity: float,
    ytm: float,
    freq: int = 2,
) -> float:
    """Bond convexity: second-order price sensitivity to yield.

    Convexity is always positive for standard bonds (no embedded options).
    Higher convexity = bond price benefits more from rate moves in
    either direction (favorable asymmetry).
    """
    n_periods = int(maturity * freq)
    coupon = face * coupon_rate / freq
    y = ytm / freq
    price = bond_price(coupon_rate, face, maturity, ytm, freq)

    if price <= 0:
        return 0.0

    conv_sum = 0.0
    for t in range(1, n_periods + 1):
        cf = coupon if t < n_periods else coupon + face
        pv = cf / (1 + y) ** (t + 2)
        conv_sum += t * (t + 1) * pv

    return float(conv_sum / (price * freq**2))


def dv01(
    coupon_rate: float,
    face: float,
    maturity: float,
    ytm: float,
    freq: int = 2,
) -> float:
    """Dollar Value of a 01 — price change per 1bp yield move.

    DV01 = modified_duration * price * 0.0001

    This is the standard risk metric for fixed income portfolios.
    A DV01 of $50 means the position gains/loses $50 per 1bp move.
    """
    mod_dur = modified_duration(coupon_rate, face, maturity, ytm, freq)
    price = bond_price(coupon_rate, face, maturity, ytm, freq)
    return float(mod_dur * price * 0.0001)


def price_change_estimate(
    mod_dur: float,
    conv: float,
    dy: float,
) -> float:
    """Estimate price change using duration + convexity approximation.

    dP/P ≈ -D_mod * dy + 0.5 * C * dy²

    Parameters
    ----------
    mod_dur : modified duration
    conv : convexity
    dy : yield change (e.g., 0.01 = 100bp increase)
    """
    return float(-mod_dur * dy + 0.5 * conv * dy**2)


def compute_analytics(
    coupon_rate: float,
    face: float = 100.0,
    maturity: float = 10.0,
    ytm: float = 0.05,
    freq: int = 2,
) -> BondAnalytics:
    """Compute all bond analytics in one call."""
    return BondAnalytics(
        price=bond_price(coupon_rate, face, maturity, ytm, freq),
        ytm=ytm,
        macaulay_duration=macaulay_duration(coupon_rate, face, maturity, ytm, freq),
        modified_duration=modified_duration(coupon_rate, face, maturity, ytm, freq),
        convexity=convexity(coupon_rate, face, maturity, ytm, freq),
        dv01=dv01(coupon_rate, face, maturity, ytm, freq),
    )
