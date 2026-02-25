import numpy as np


def report_noisy_max(u: np.ndarray, gs: float, eps: float) -> int:
    """Report noisy max selection mechanism.

    Parameters
    ----------
    u : np.ndarray
        Utility scores for each candidate.
    gs : float
        Global sensitivity of the utility function.
    eps : float
        Privacy budget.

    Returns
    -------
    int
        Index of the selected candidate.
    """

    lbd = eps / (2 * gs)
    noisy_u = u + np.random.exponential(1 / lbd, size=u.size)
    return int(np.argmax(noisy_u))


# def safe_exp(x: float) -> float:
#     return float(decimal.Decimal(x).exp())


def safe_exp(x: float) -> float:
    MAX_EXP = 680
    if x > MAX_EXP:
        return np.exp(MAX_EXP)
    elif x < -MAX_EXP:
        return np.exp(-MAX_EXP)
    else:
        return np.exp(x)


def em_pmf(u: np.ndarray, gs: float, eps: float):
    u = u - u.max()
    scaled_u = np.array([safe_exp((r * eps) / (2 * gs)) for r in u])
    scores = scaled_u / scaled_u.sum()
    return scores


def em(u: np.ndarray, gs: float, eps: float) -> int:
    scale = (2 * gs) / eps
    noisy_u = u + np.random.gumbel(0, scale, size=u.size)
    return int(np.argmax(noisy_u))
