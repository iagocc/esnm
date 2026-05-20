import math


def rho_zcdp_to_eps_for_pure_dp(rho: float) -> float:
    """
    Given a zCDP budget rho, return the pure-DP epsilon such that
    an epsilon-DP mechanism is rho-zCDP.

    Applies to genuinely pure-(epsilon, 0)-DP mechanisms — the exponential
    mechanism, report-noisy-max, ShiftedInverse, and Student's-T smooth
    sensitivity noise (Bun & Steinke 2019, Thm 31; Cauchy / polynomial tails) —
    which satisfy (1/2 epsilon^2)-zCDP by Bun & Steinke 2016.

    Uses rho = epsilon^2 / 2, hence epsilon = sqrt(2 rho).

    NOTE: for noise distributions that are NOT pure-DP (Laplace log-normal etc.)
    use `rho_zcdp_to_cdp_eps` instead — it returns the same value but for the
    correct (CDP-native) reason.
    """
    if rho < 0:
        raise ValueError("rho must be non-negative.")

    return math.sqrt(2.0 * rho)


def rho_zcdp_to_cdp_eps(rho: float) -> float:
    """
    Given a zCDP budget rho, return the CDP parameter epsilon of a
    (1/2 epsilon^2)-CDP mechanism so that it is exactly rho-zCDP.

    The smooth-sensitivity noise distributions of Bun & Steinke 2019 that are
    NOT pure-DP — Laplace log-normal (Prop. 3), uniform log-normal (Prop. 4),
    arsinh-normal (Thm 25) — are directly (1/2 epsilon^2)-CDP. Exponential /
    quasi-polynomial tails cannot give pure DP, so these never go through the
    pure-DP route. Concentrated DP with parameter (1/2 epsilon^2) is identical
    to rho-zCDP with rho = epsilon^2 / 2, hence epsilon = sqrt(2 rho).

    Numerically identical to `rho_zcdp_to_eps_for_pure_dp` (both sqrt(2 rho));
    kept separate because the privacy justification differs.
    """
    if rho < 0:
        raise ValueError("rho must be non-negative.")

    return math.sqrt(2.0 * rho)


def rho_zcdp_to_eps_delta_dp(rho: float, delta: float) -> float:
    """
    Convert rho-zCDP to (epsilon, delta)-DP using the standard bound:

        epsilon = rho + 2 * sqrt(rho * log(1 / delta))

    Valid for rho >= 0 and 0 < delta < 1.
    """
    if rho < 0:
        raise ValueError("rho must be non-negative.")
    if not (0 < delta < 1):
        raise ValueError("delta must satisfy 0 < delta < 1.")

    if rho == 0:
        return 0.0

    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
