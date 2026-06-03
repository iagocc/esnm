# eSNM - Element Smooth Noisy Max

A high-performance implementation of the **Element Smooth Noisy Max (eSNM)** mechanism for differentially private selection. This project combines C++ performance-critical components (via [nanobind](https://github.com/wjakob/nanobind)) with Python experiment harnesses to evaluate eSNM across multiple application domains.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Running the Experiments](#running-the-experiments)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Overview

eSNM introduces several noise-based variants for private selection:

- **eSNM-T** -- Uses Student's t-distribution noise, with per-element smooth sensitivity scaling.
- **eSNM-GCP** -- Uses Gaussian-core Pareto-tail noise.
- **eSNM-LCP** -- Uses Laplace-core Pareto-tail noise.

All variants automatically calibrate noise on a per-element basis using smooth sensitivity, improving utility over global-sensitivity methods such as Report Noisy Max and the Exponential Mechanism.

The repository includes two experiment suites that reproduce the paper's empirical evaluation:

| Experiment | Script | Domain |
|---|---|---|
| Top-K Selection | `src/exp_topk.py` | Count-based rankings (games, books, movies) |
| Percentile Estimation | `src/exp_percentile.py` | High-percentile queries on 1-D data |

## System Requirements

### Build toolchain

| Dependency | Version |
|---|---|
| CMake | 3.15 -- 3.26 |
| C++ compiler with C++20 support | Clang (recommended) or GCC |
| Python | >= 3.13.5 |

### Native libraries

| Library | Purpose |
|---|---|
| [GSL](https://www.gnu.org/software/gsl/) (GNU Scientific Library) | Numerical integration, statistical distributions |
| [OpenMP](https://www.openmp.org/) (`libomp`) | Parallel sampling in C++ mechanisms |

**macOS (Homebrew):**

```bash
brew install gsl libomp cmake
```

**Ubuntu / Debian:**

```bash
sudo apt-get install libgsl-dev libomp-dev cmake
```

### Python dependencies

All Python dependencies are declared in `pyproject.toml` and installed automatically. Key packages include: `numba`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `networkx`, `matplotlib`, and `nanobind`.

## Installation

2. **Install dependencies and the package in editable mode:**

```bash
uv sync
uv pip install --no-build-isolation -ve .
```

> To enable automatic rebuild when C++ source files are edited, add the flag:
> ```bash
> uv pip install --no-build-isolation -ve . -Ceditable.rebuild=true
> ```

3. **Verify the installation:**

```python
uv run python -c "from esnm.mechanism import esnm_t_pmf; from esnm.percentile import get_ls; print('Build OK')"
```

## Running the Experiments

All experiment scripts are located under `src/` and should be run from the **repository root** directory, since they resolve dataset paths relative to `Path.cwd()`.

```bash
# Top-K Selection (games, books, movies)
python src/exp_topk.py

# Percentile Estimation (hepth, income, patent)
python src/exp_percentile.py
```

Results are written to the corresponding subdirectory under `results/` (e.g., `results/topk/`, `results/percentile/`).

### Experiment details

**`exp_topk.py`** -- Compares the joint exponential mechanism baseline against eSNM-T, eSNM-GCP, and eSNM-LCP for top-k count selection. All methods are pure eps-DP and receive the budget directly (no zCDP conversion). Reports L1 error, L-infinity error, and NDCG across varying k values at a pure-DP epsilon = 1.0 over 50 trials.

**`exp_percentile.py`** -- Benchmarks eSNM-T, eSNM-GCP, and eSNM-LCP against Local Dampening, Shifted Local Dampening, and ShiftedInverse for privately estimating percentiles on 1-D datasets, over a pure eps-DP budget sweep. Uses precomputed local sensitivity matrices stored in `states/`.

## Project Structure

```
eSNM/
├── CMakeLists.txt                  # Build configuration (C++20, nanobind, GSL, OpenMP)
├── pyproject.toml                  # Python project metadata and dependencies
├── README.md
│
├── src/
│   ├── mechanism.cpp               # Core eSNM mechanisms (C++, nanobind bindings)
│   │
│   ├── distributions/
│   │   ├── gcp.h                   # Gaussian-core Pareto-tail noise (PDF/survival)
│   │   ├── lcp.h                   # Laplace-core Pareto-tail noise (PDF/survival)
│   │   └── prng.h / prng.cpp       # xoshiro256++ PRNG
│   │
│   ├── esnm/                       # Python package (compiled modules installed here)
│   │   └── __init__.py
│   │
│   ├── optimize_params.py          # Numba-accelerated parameter optimization
│   ├── standard_selection.py       # Baseline mechanisms (Report Noisy Max, EM)
│   ├── local_dampening.py          # Local Dampening mechanism
│   │
│   ├── shifted_inverse.py          # ShiftedInverse percentile sampler (baseline)
│   │
│   ├── topk/                       # Top-K selection module
│   │   ├── esnm_joint.py           # eSNM adapted for joint top-k
│   │   └── joint.py                # Joint exponential mechanism
│   │
│   ├── percentile/
│   │   └── percentile.cpp          # Percentile computation (C++, nanobind)
│   │
│   ├── exp_topk.py                 # Experiment: Top-K selection
│   └── exp_percentile.py           # Experiment: Percentile estimation
│
├── data/
│   ├── topk/                       # Count datasets (games, books, movies)
│   └── 1D/                         # 1-D numeric arrays for percentile experiments
│
├── states/                         # Precomputed local sensitivity matrices (.npy)
├── results/                        # Experiment output (TSV files per experiment)
│   ├── topk/
│   └── percentile/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{esnm2025,
  title   = {TODO: Paper Title},
  author  = {TODO: Authors},
  journal = {TODO: Journal/Conference},
  year    = {TODO},
}
```

## License

MIT
