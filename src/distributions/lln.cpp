#include "lln.h"

#include <gsl/gsl_poly.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace nb = nanobind;

std::tuple<double, double> get_s_and_sigma(double eps, double t) {
    double opt_sigma, x1, x2;
    int res = gsl_poly_solve_cubic(-t / eps, 0, -t / (5 * eps),
                                   &opt_sigma, &x1, &x2);
    if (res < 1)
        std::runtime_error("Can't found the roots for sigma optimization.");

    double s = std::exp(-(3.0 / 2) * std::pow(opt_sigma, 2 * (eps - t / opt_sigma)));
    return {s, opt_sigma};
}

double lln_optimize_t(std::vector<double> ts, std::vector<double> smooths, double eps) {
    double min_v   = std::numeric_limits<double>::max();
    double best_t  = std::numeric_limits<double>::max();

    for (int i = 0; i < static_cast<int>(ts.size()); i++) {
        auto [s, sigma] = get_s_and_sigma(eps, ts[i]);
        double var = (2 * std::exp(2 * std::pow(sigma, 2)) * std::pow(smooths[i], 2))
                     / std::pow(s, 2);
        if (var < min_v) {
            best_t = ts[i];
            min_v  = var;
        }
    }

    return best_t;
}

NB_MODULE(dist_lln, m) {
    m.def("lln_pdf", &lln_pdf);
    m.def("lln_cdf", &lln_cdf);
    m.def("get_s_and_sigma", &get_s_and_sigma);
}
