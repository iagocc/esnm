#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <tgmath.h>
#include <stdexcept>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;
using namespace nb::literals;

using Dataset = std::vector<double>;
using DatasetList = std::vector<Dataset>;

Dataset copyAndReplace(const Dataset& x, size_t index, double value) {
    if (index >= x.size()) {
        throw std::out_of_range("Index is out of range in copyAndReplace.");
    }
    Dataset x_prime = x;
    x_prime[index] = value;
    std::sort(x_prime.begin(), x_prime.end());
    return x_prime;
}

DatasetList candidates(const Dataset& x, size_t t, size_t r_idx, double p, double cap_lambda, const DatasetList& prev_candidates) {
    const size_t n = x.size();
    if (n == 0) return {};

    if (r_idx >= n) {
        throw std::out_of_range("Range element 'r_idx' must be a valid 0-based index.");
    }
    
    if (t == 0) return {x};

    const size_t k_idx = static_cast<size_t>(std::ceil(p*(n + 1))) - 1;
    if (k_idx >= n) {
         throw std::logic_error("Calculated median index 'k' is out of bounds.");
    }

    if (t == 1) {
        DatasetList result;
        const Dataset xp2 = x, xp4 = x;
        result.reserve(6);
        result.push_back(copyAndReplace(x, r_idx, cap_lambda)); // x'_1
        result.push_back(xp2);                                // x'_2
        result.push_back(copyAndReplace(x, r_idx, 0.0));    // x'_3
        result.push_back(xp4);                                // x'_4
        result.push_back(copyAndReplace(x, k_idx, cap_lambda)); // x'_5
        result.push_back(copyAndReplace(x, k_idx, 0.0));    // x'_6
        return result;
    }

    // DatasetList prev_candidates = candidates(x, t - 1, r_idx, p, cap_lambda);
    
    if (prev_candidates.size() != 6) {
        throw std::logic_error("Recursive call did not return 6 datasets as expected.");
    }
    
    const auto& x1 = prev_candidates.at(0);
    const auto& x2 = prev_candidates.at(1);
    const auto& x3 = prev_candidates.at(2);
    const auto& x4 = prev_candidates.at(3);
    const auto& x5 = prev_candidates.at(4);
    const auto& x6 = prev_candidates.at(5);

    DatasetList result;
    result.reserve(6);
    result.push_back(copyAndReplace(x1, k_idx, 0.0));    // x'_1
    result.push_back(copyAndReplace(x2, k_idx, cap_lambda)); // x'_2
    result.push_back(copyAndReplace(x3, k_idx, 0.0));    // x'_3
    result.push_back(copyAndReplace(x4, k_idx, cap_lambda)); // x'_4
    result.push_back(copyAndReplace(x5, k_idx, 0.0));    // x'_5
    result.push_back(copyAndReplace(x6, k_idx, cap_lambda)); // x'_6

    return result;
}

double ls_distance_0(const Dataset& x, size_t i, double p, double cap_lambda) {
    const size_t n = x.size();
    if (n == 0) return 0.0;

    if (i > n) {
        throw std::out_of_range("Index 'i' must be a valid 1-based index for ls_distance_0.");
    }
    const size_t i_idx = i;

    const size_t k = static_cast<size_t>(std::ceil(p*(n + 1)));
    const size_t k_idx = k - 1;

    const double val_i = x[i_idx];
    const double val_k = x[k_idx];
    const double val_k_minus_1 = (k_idx > 0) ? x[k_idx - 1] : 0.0;
    const double val_k_plus_1 = (k_idx < n - 1) ? x[k_idx + 1] : cap_lambda;

    double term1 = std::abs(val_k - val_i);

    double term2 = std::abs(std::abs(val_k - val_i) - std::abs(val_k_plus_1 - val_i));//val_k_plus_1 - val_k;

    double term3 = std::abs(std::abs(val_k - val_i) - std::abs(val_k_minus_1 - val_i));//val_k - val_k_minus_1;

    double p_val;
    double q_val;
    if (i > k) {
        p_val = cap_lambda - val_i;
        q_val = std::abs(val_i - val_k - val_k_minus_1);//val_i;
    } else if (i == k) {
        p_val = cap_lambda - val_k_plus_1;
        q_val = val_k_minus_1;
    } else { // i < k
        p_val = std::abs(val_k - val_i - cap_lambda + val_k_plus_1);//cap_lambda + val_i - 3 * val_k + val_k_plus_1;
        q_val = val_i;//3 * val_k - val_i - val_k_minus_1;
    }

    return std::max({term1, term2, term3, p_val, q_val});
}

std::tuple<double, DatasetList> ls_distance_t(const Dataset& x, size_t t, size_t r_idx, double p, double cap_lambda, DatasetList& prev_cand) {
    DatasetList candidate_datasets;

    if (t == 0) {
        candidate_datasets = candidates(x, t, r_idx, p, cap_lambda, DatasetList());
    } else {
        candidate_datasets = candidates(x, t, r_idx, p, cap_lambda, prev_cand);
    }

    if (candidate_datasets.empty()) {
        return std::make_tuple(0.0, DatasetList());
    }

    double max_ls = 0.0;

    for (const auto& y : candidate_datasets) {
        double current_ls = ls_distance_0(y, r_idx, p, cap_lambda);
        
        if (current_ls > max_ls) {
            max_ls = current_ls;
        }
    }
    return std::make_tuple(max_ls, candidate_datasets);
}

std::vector<std::vector<double>> get_ls(const Dataset& x, double p, double cap_lambda) {
    size_t n = x.size();
    std::vector<std::vector<double>> ls(n, std::vector<double>(n));

    for(size_t r = 0; r < n; r++) {
        printf("Getting LS from r = %zu\n", r);
        DatasetList prev = DatasetList();
        for(size_t t = 0; t < n; t++) {
            double sens;
            std::tie(sens, prev) = ls_distance_t(x, t, r, p, cap_lambda, prev);
            ls[r][t] = sens;
        }
    }

    return ls;
}

double smooth_s(const Dataset& x, size_t r, double p, double t, size_t max_iter, double cap_lambda, double gs) {
    double sens = 0, ls_t = 0;
    double lower_bound = static_cast<double>(static_cast<long double>(gs) * std::expl(static_cast<long double>(-t * (max_iter+1))));
    DatasetList prev = DatasetList();

    for(size_t k = 0; k < max_iter; k++) {
        long double exp_fact = std::expl(static_cast<long double>(-t * k));
        std::tie(ls_t, prev) = ls_distance_t(x, k, r, p, cap_lambda, prev);
        double ls = static_cast<double>(static_cast<long double>(ls_t) * exp_fact);
        sens = std::max({sens, ls});
    }

    return std::min({gs, std::max({sens, lower_bound})});
}


NB_MODULE(percentile, m) {
  m.def("candidates", &candidates);
  m.def("get_ls", &get_ls);
  m.def("ls_distance_t", &ls_distance_t, "x"_a, "t"_a, "r_idx"_a, "p"_a, "cap_lambda"_a, "last_ls"_a);
  m.def("ls_distance_0", &ls_distance_0);
  m.def("smooth_s", &smooth_s);
}
    
