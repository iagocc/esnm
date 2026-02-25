#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <limits>
#include <numeric>
#include <vector>

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_randist.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <omp.h>

#include "distributions/lln.h"
#include "distributions/prng.h"

namespace nb = nanobind;

using Utilities = std::vector<double>;
using Probs     = std::vector<double>;

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

struct TopTwoIndices {
    size_t max_idx;
    size_t second_idx;
};

// O(n) scan — replaces the previous O(n log n) full sort that only needed
// the two largest elements.
inline TopTwoIndices top_two_utility_indices(const Utilities& u) {
    assert(!u.empty());

    size_t max_idx    = 0;
    size_t second_idx = 0;  // equals max_idx when n == 1 (intentional)
    for (size_t i = 1; i < u.size(); ++i) {
        if (u[i] > u[max_idx]) {
            second_idx = max_idx;
            max_idx    = i;
        } else if (second_idx == max_idx || u[i] > u[second_idx]) {
            second_idx = i;
        }
    }

    return {max_idx, second_idx};
}

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------

struct TStudentDistribution {
    double df;
    double pdf(double x, double /*sigma*/) const {
        return gsl_ran_tdist_pdf(x, df);
    }
    double survival(double x, double /*sigma*/) const {
        return gsl_cdf_tdist_Q(x, df);
    }
};

struct LLNDistribution {
    static double pdf(double x, double sigma)      { return lln_pdf(x, sigma); }
    static double survival(double x, double sigma) { return 1.0 - lln_cdf(x, sigma); }
};

// ---------------------------------------------------------------------------
// Noise samplers
// ---------------------------------------------------------------------------

inline double sample_t_student_noise(gsl_rng* r, double df) {
    const double z = gsl_ran_gaussian_ziggurat(r, 1.0);
    const double v = gsl_ran_chisq(r, df);
    return z / std::sqrt(v / df);
}

inline double sample_lln_noise(gsl_rng* r, double sigma) {
    const double x = gsl_ran_laplace(r, 1.0);
    const double y = gsl_ran_gaussian_ziggurat(r, 1.0);
    return x * std::exp(sigma * y);
}

// ---------------------------------------------------------------------------
// Argmax sampler
// ---------------------------------------------------------------------------

template <typename NoiseSampler>
size_t esnm_sample_argmax(
    const Utilities&           u,
    const std::vector<double>& smooth_s,
    const std::vector<double>& s_values,
    NoiseSampler               noise_sampler
) {
    const size_t n = u.size();
    const TopTwoIndices top_two        = top_two_utility_indices(u);
    const double        smooth_s_max   = smooth_s[top_two.max_idx];
    const double        smooth_s_second = smooth_s[top_two.second_idx];

    size_t selected_idx     = 0;
    double selected_utility = -std::numeric_limits<double>::infinity();

    #pragma omp parallel
    {
        gsl_rng* r = gsl_rng_alloc(gsl_rng_xoshiro256plusplus);
        const unsigned long seed =
            static_cast<unsigned long>(std::time(nullptr))
            ^ (static_cast<unsigned long>(omp_get_thread_num()) * 1099511628211ULL);
        gsl_rng_set(r, seed);

        double local_best_utility = -std::numeric_limits<double>::infinity();
        size_t local_best_idx     = 0;

        #pragma omp for nowait
        for (size_t i = 0; i < n; ++i) {
            const double noise            = noise_sampler(r, i);
            const double smooth_s_r_star  = (top_two.max_idx == i) ? smooth_s_second
                                                                     : smooth_s_max;
            const double sensitivity      = smooth_s[i] + smooth_s_r_star;
            const double scale            = sensitivity / s_values[i];
            const double noisy_utility    = u[i] + noise * scale;

            if (noisy_utility > local_best_utility ||
                (noisy_utility == local_best_utility && i < local_best_idx)) {
                local_best_utility = noisy_utility;
                local_best_idx     = i;
            }
        }

        #pragma omp critical(esnm_select)
        {
            if (local_best_utility > selected_utility ||
                (local_best_utility == selected_utility && local_best_idx < selected_idx)) {
                selected_utility = local_best_utility;
                selected_idx     = local_best_idx;
            }
        }

        gsl_rng_free(r);
    }

    return selected_idx;
}

// ---------------------------------------------------------------------------
// PMF integrand
// ---------------------------------------------------------------------------

struct integrand_params {
    const Utilities*           u;
    const std::vector<double>* sens;
    size_t                     r;
    const std::vector<double>* s_values;
    const std::vector<double>* sigmas;
    double                     df;
    // Pre-computed values to avoid sorting inside the integrand
    double sens_r_star;  // smooth_s[r_star] for this element
    double u_r_star;     // u[r_star] for this element
};

template <typename Distribution>
double generic_integrand(double x, void* params, const Distribution& dist) {
    const integrand_params& p  = *static_cast<integrand_params*>(params);
    const size_t            r  = p.r;
    const double sensitivity   = p.sens->at(r) + p.sens_r_star;
    const double scale         = sensitivity / p.s_values->at(r);
    const double sigma         = p.sigmas ? p.sigmas->at(r) : 0.0;
    const double y             = dist.pdf(x, sigma);
    const double scaled_diff   = (p.u_r_star - p.u->at(r) + x) / scale;
    const double survival      = dist.survival(scaled_diff, sigma);
    return y * survival;
}

double t_student_integrand(double x, void* params) {
    const integrand_params& p = *static_cast<integrand_params*>(params);
    TStudentDistribution dist{p.df};
    return generic_integrand(x, params, dist);
}
double lln_integrand(double x, void* params) {
    LLNDistribution dist;
    return generic_integrand(x, params, dist);
}

using Integrand = double (*)(double, void*);

// ---------------------------------------------------------------------------
// PMF computation
// ---------------------------------------------------------------------------

Probs compute_pmf(
    const Utilities&           u,
    const std::vector<double>& smooth_s,
    const std::vector<double>& s_values,
    const std::vector<double>* sigmas,
    double                     df,
    Integrand                  integrand,
    const char*                mechanism_name,
    const std::vector<size_t>* R      = nullptr,
    double                     epsabs = 0,
    double                     epsrel = 1e-7
) {
    const size_t n      = u.size();
    const size_t R_size = (R == nullptr) ? n : R->size();
    std::vector<double> p(R_size);

    // O(n) scan for top-two — replaces previous O(n log n) full sort.
    // When n == 1, second_idx == max_idx, giving the correct fallback values.
    const TopTwoIndices top_two     = top_two_utility_indices(u);
    const double        sens_max    = smooth_s[top_two.max_idx];
    const double        sens_second = smooth_s[top_two.second_idx];
    const double        u_max       = u[top_two.max_idx];
    const double        u_second    = u[top_two.second_idx];

    #pragma omp parallel for
    for (size_t i = 0; i < R_size; i++) {
        // When R is provided, use the actual element index; otherwise i itself.
        const size_t r = (R == nullptr) ? i : R->at(i);

        gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);

        const bool   is_max      = (top_two.max_idx == r);
        const double sens_r_star = is_max ? sens_second : sens_max;
        const double u_r_star    = is_max ? u_second    : u_max;

        integrand_params params;
        params.u           = &u;
        params.sens        = &smooth_s;
        params.r           = r;
        params.s_values    = &s_values;
        params.sigmas      = sigmas;
        params.df          = df;
        params.sens_r_star = sens_r_star;
        params.u_r_star    = u_r_star;

        gsl_function F;
        F.function = integrand;
        F.params   = &params;

        double     result, error;
        const int  status = gsl_integration_qagi(&F, epsabs, epsrel, 1000, w,
                                                  &result, &error);
        if (status != GSL_SUCCESS) {
            std::fprintf(stderr,
                         "GSL integration for %s Mechanism failed: %s\n",
                         mechanism_name, gsl_strerror(status));
        }

        p[i] = result;
        gsl_integration_workspace_free(w);
    }

    return p;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

Probs esnm_t_pmf(
    const Utilities&           u,
    const std::vector<double>& smooth_s,
    const std::vector<double>& s_values,
    double                     df,
    const std::vector<size_t>* R = nullptr
) {
    return compute_pmf(u, smooth_s, s_values, nullptr, df,
                       &t_student_integrand, "T-Student", R, 0, 1e-12);
}

Probs esnm_lln_pmf(
    const Utilities&           u,
    const std::vector<double>& smooth_s,
    const std::vector<double>& s_values,
    const std::vector<double>& sigmas,
    const std::vector<size_t>* R = nullptr
) {
    return compute_pmf(u, smooth_s, s_values, &sigmas, 0,
                       &lln_integrand, "LLN", R);
}

size_t esnm_lln(
    const Utilities&           u,
    const std::vector<double>& smooth_s,
    const std::vector<double>& s_values,
    const std::vector<double>& sigmas
) {
    return esnm_sample_argmax(u, smooth_s, s_values,
                              [&sigmas](gsl_rng* r, size_t i) {
                                  return sample_lln_noise(r, sigmas[i]);
                              });
}

size_t esnm_t(
    const Utilities&           u,
    const std::vector<double>& smooth_s,
    const std::vector<double>& s_values,
    double                     df
) {
    return esnm_sample_argmax(u, smooth_s, s_values,
                              [df](gsl_rng* r, size_t /*i*/) {
                                  return sample_t_student_noise(r, df);
                              });
}

NB_MODULE(mechanism, m) {
    m.def("esnm_t_pmf",
          [](const Utilities& u, const std::vector<double>& smooth_s,
             const std::vector<double>& s_values, double df) -> Probs {
              return esnm_t_pmf(u, smooth_s, s_values, df, nullptr);
          });
    m.def("esnm_lln_pmf",
          [](const Utilities& u, const std::vector<double>& smooth_s,
             const std::vector<double>& s_values,
             const std::vector<double>& sigmas) -> Probs {
              return esnm_lln_pmf(u, smooth_s, s_values, sigmas, nullptr);
          });
    m.def("esnm_lln", &esnm_lln);
    m.def("esnm_t",   &esnm_t);
}
