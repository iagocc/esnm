#pragma once

#include <cmath>

// ---------------------------------------------------------------------------
// GCP (Gaussian-core Pareto-tail) PDF and survival
// ---------------------------------------------------------------------------
// Symmetric distribution with a Gaussian body for |z| <= z0 spliced C^1 to a
// polynomial (Pareto) tail for |z| > z0. The C^1 splice forces z0 = sigma·√(γ+1).
// All closed form — no numerical integration needed.
// ---------------------------------------------------------------------------

namespace gcp_detail {

// Standard normal CDF.
static inline double phi(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Normalizer constant κ(γ) (the normalizer C = sigma·κ(γ)).
static inline double kappa(double gamma) {
    const double g1 = gamma + 1.0;
    const double sg1 = std::sqrt(g1);
    return std::sqrt(2.0 * M_PI) * (2.0 * phi(sg1) - 1.0)
         + (2.0 / gamma) * sg1 * std::exp(-0.5 * g1);
}

} // namespace gcp_detail

inline double gcp_pdf(double z, double sigma, double gamma) {
    const double g1   = gamma + 1.0;
    const double z0   = sigma * std::sqrt(g1);
    const double C    = sigma * gcp_detail::kappa(gamma);
    const double az   = std::fabs(z);

    if (az <= z0)
        return std::exp(-z * z / (2.0 * sigma * sigma)) / C;

    return std::exp(-0.5 * g1) * std::pow(z0 / az, g1) / C;
}

inline double gcp_survival(double z, double sigma, double gamma) {
    if (z < 0.0) return 1.0 - gcp_survival(-z, sigma, gamma);

    const double g1  = gamma + 1.0;
    const double z0  = sigma * std::sqrt(g1);
    const double C   = sigma * gcp_detail::kappa(gamma);

    if (z <= z0) {
        return (sigma * std::sqrt(2.0 * M_PI)
                    * (gcp_detail::phi(z0 / sigma) - gcp_detail::phi(z / sigma))
                + std::exp(-0.5 * g1) * z0 / gamma) / C;
    }

    return std::exp(-0.5 * g1) * z0 * std::pow(z0 / z, gamma) / (gamma * C);
}
