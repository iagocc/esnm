import math


def rho_zcdp_to_eps_for_pure_dp(rho: float) -> float:
    """
    Given a zCDP budget rho, return the pure-DP epsilon such that
    an epsilon-DP mechanism, e.g. the exponential mechanism, is rho-zCDP.

    Uses:
        rho = epsilon^2 / 2
    hence:
        epsilon = sqrt(2 rho)
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
