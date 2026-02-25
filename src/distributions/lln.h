#pragma once

#include <cmath>
#include <cstdio>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

// ---------------------------------------------------------------------------
// LLN PDF
// ---------------------------------------------------------------------------

namespace lln_detail {

struct PdfParams {
    double z;
    double sigma;
};

static inline double pdf_integrand(double y, void* vp) {
    const PdfParams* p = static_cast<PdfParams*>(vp);
    const double exponent =
        -0.5 * y * y - p->z * std::exp(-p->sigma * y) - p->sigma * y;
    static const double norm = 1.0 / (2.0 * std::sqrt(2.0 * M_PI));
    return norm * std::exp(exponent);
}

struct CdfParams {
    double z;
    double sigma;
};

static inline double cdf_integrand(double y, void* vp) {
    const CdfParams* p = static_cast<CdfParams*>(vp);
    const double exp_term = std::exp(-p->z * std::exp(-p->sigma * y));
    const double gauss    = std::exp(-0.5 * y * y) / std::sqrt(2.0 * M_PI);
    return gauss * exp_term;
}

} // namespace lln_detail

inline double lln_pdf(double z, double sigma) {
    if (z == 0.0)
        return std::exp(sigma * sigma / 2.0) / 2.0;

    gsl_integration_workspace* w = gsl_integration_workspace_alloc(10000);

    lln_detail::PdfParams params{std::fabs(z), sigma};
    gsl_function F{&lln_detail::pdf_integrand, &params};

    double result, error;
    const int status = gsl_integration_qagi(&F, 1e-9, 1e-7, 10000, w, &result, &error);
    gsl_integration_workspace_free(w);

    if (status != GSL_SUCCESS) {
        std::fprintf(stderr, "GSL integration for LLN PDF failed: %s\n",
                     gsl_strerror(status));
        return NAN;
    }
    return result;
}

inline double lln_cdf(double z, double sigma) {
    if (z == 0.0) return 0.5;
    if (z < 0.0)  return 1.0 - lln_cdf(-z, sigma);

    gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);

    lln_detail::CdfParams params{z, sigma};
    gsl_function F{&lln_detail::cdf_integrand, &params};

    double result, error;
    const int status = gsl_integration_qagi(&F, 1e-9, 1e-7, 1000, w, &result, &error);

    if (status != GSL_SUCCESS) {
        std::fprintf(stderr, "GSL integration for LLN CDF failed: %s\n",
                     gsl_strerror(status));
        gsl_integration_workspace_free(w);
        return NAN;
    }

    gsl_integration_workspace_free(w);

    double Fz = 1.0 - 0.5 * result;
    if (Fz < 0.0) Fz = 0.0;
    if (Fz > 1.0) Fz = 1.0;
    return Fz;
}
