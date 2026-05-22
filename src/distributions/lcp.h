#pragma once

#include <cmath>

// ---------------------------------------------------------------------------
// LCP (Laplace-core Pareto-tail) PDF and survival
// ---------------------------------------------------------------------------
// Symmetric distribution with a Laplace body for |z| <= z0 spliced C^1 to a
// polynomial (Pareto) tail for |z| > z0. The C^1 splice forces
// z0 = sigma * (gamma + 1). All closed form -- no numerical integration needed.
// ---------------------------------------------------------------------------

namespace lcp_detail {

// Normalizer constant kappa(gamma) (the normalizer C = 2 * sigma * kappa).
static inline double kappa(double gamma) {
    const double g1       = gamma + 1.0;
    const double exp_term = std::exp(-g1);
    return 1.0 - exp_term + g1 * exp_term / gamma;
}

} // namespace lcp_detail

inline double lcp_pdf(double z, double sigma, double gamma) {
    const double g1       = gamma + 1.0;
    const double z0       = sigma * g1;
    const double C        = 2.0 * sigma * lcp_detail::kappa(gamma);
    const double az       = std::fabs(z);
    const double exp_term = std::exp(-g1);

    if (az <= z0)
        return std::exp(-az / sigma) / C;

    return exp_term * std::pow(z0 / az, g1) / C;
}

inline double lcp_survival(double z, double sigma, double gamma) {
    if (z < 0.0) return 1.0 - lcp_survival(-z, sigma, gamma);

    const double g1       = gamma + 1.0;
    const double z0       = sigma * g1;
    const double C        = 2.0 * sigma * lcp_detail::kappa(gamma);
    const double exp_term = std::exp(-g1);

    if (z <= z0) {
        return (sigma * (std::exp(-z / sigma) - exp_term)
                + exp_term * z0 / gamma) / C;
    }

    return exp_term * z0 * std::pow(z0 / z, gamma) / (gamma * C);
}
