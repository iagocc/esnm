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

eSNM introduces two noise-based variants for private selection:

- **eSNM-T** -- Uses Student's t-distribution noise, with per-element smooth sensitivity scaling.
- **eSNM-LLN** -- Uses Log-Laplace noise, offering robustness for elements with extreme local sensitivities.

Both variants automatically calibrate noise on a per-element basis using smooth sensitivity, improving utility over global-sensitivity methods such as Report Noisy Max and the Exponential Mechanism.

The repository includes four experiment suites that reproduce the paper's empirical evaluation:

| Experiment | Script | Domain |
|---|---|---|
| Top-K Selection | `src/exp_topk.py` | Count-based rankings (games, books, movies) |
| Egocentric Betweenness Centrality | `src/exp_ebc.py` | Influential node identification in social networks |
| Percentile Estimation | `src/exp_percentile.py` | High-percentile queries on 1-D data |
| Private Decision Trees | `src/exp_tree.py` | Differentially private ID3 attribute selection |

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
uv run python -c "from esnm.mechanism import esnm_t, esnm_lln; print('Build OK')"
```

## Running the Experiments

All experiment scripts are located under `src/` and should be run from the **repository root** directory, since they resolve dataset paths relative to `Path.cwd()`.

```bash
# Top-K Selection (games, books, movies)
python src/exp_topk.py

# Egocentric Betweenness Centrality (enron, dblp, github)
python src/exp_ebc.py

# Percentile Estimation (hepth, income, patent)
python src/exp_percentile.py

# Private Decision Trees (adult, nltcs, acs)
python src/exp_tree.py
```

Results are written to the corresponding subdirectory under `results/` (e.g., `results/topk/`, `results/ebc/`).

### Experiment details

**`exp_topk.py`** -- Compares the joint exponential mechanism baseline against eSNM-T and eSNM-LLN for top-k count selection. Reports L1 error, L-infinity error, and NDCG across varying k values at epsilon = 1.0 over 50 trials.

**`exp_ebc.py`** -- Evaluates multiple differentially private mechanisms (Report Noisy Max, Local Dampening, Shifted Local Dampening, eSNM-T, eSNM-LLN) for identifying the top-k most influential nodes in social network graphs. Sweeps epsilon in logspace and reports accuracy (overlap with the true top-k).

**`exp_percentile.py`** -- Tests eSNM-T and eSNM-LLN for privately estimating high percentiles (e.g., 99th) on 1-D datasets. Uses precomputed local sensitivity matrices stored in `states/`.

**`exp_tree.py`** -- Builds differentially private ID3 decision trees using eSNM and Local Dampening for attribute selection at each split. Evaluates classification accuracy via 10-fold cross-validation with varying tree depths and privacy budgets.

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
│   │   ├── lln.h / lln.cpp         # Log-Laplace Noise distribution (PDF/CDF)
│   │   └── prng.h / prng.cpp       # xoshiro256++ PRNG
│   │
│   ├── esnm/                       # Python package (compiled modules installed here)
│   │   └── __init__.py
│   │
│   ├── optimize_params.py          # Numba-accelerated parameter optimization
│   ├── standard_selection.py       # Baseline mechanisms (Report Noisy Max, EM)
│   ├── local_dampening.py          # Local Dampening mechanism
│   │
│   ├── topk/                       # Top-K selection module
│   │   ├── esnm_joint.py           # eSNM adapted for joint top-k
│   │   ├── joint.py                # Joint exponential mechanism
│   │   └── utility.py              # Ranking utilities
│   │
│   ├── ebc/                        # Egocentric Betweenness Centrality module
│   │   ├── ebc_metric.py           # EBC computation (Numba-accelerated)
│   │   ├── graph.py                # Graph loading and adjacency utilities
│   │   └── sensitivity.py          # Local/global sensitivity for EBC
│   │
│   ├── percentile/
│   │   └── percentile.cpp          # Percentile computation (C++, nanobind)
│   │
│   ├── decision_tree/
│   │   ├── id3.py                  # Differentially private ID3 algorithm
│   │   └── candidates.py           # Local sensitivity for candidate attributes
│   │
│   ├── exp_topk.py                 # Experiment: Top-K selection
│   ├── exp_ebc.py                  # Experiment: Influential node analysis
│   ├── exp_percentile.py           # Experiment: Percentile estimation
│   └── exp_tree.py                 # Experiment: Private decision trees
│
├── data/
│   ├── graph/                      # Social network edge lists (enron, dblp, github)
│   ├── topk/                       # Count datasets (games, books, movies)
│   ├── tree/                       # Classification datasets (adult, nltcs, acs)
│   └── 1D/                         # 1-D numeric arrays for percentile experiments
│
├── states/                         # Precomputed local sensitivity matrices (.npz)
├── results/                        # Experiment output (CSV files per experiment)
│   ├── ebc/
│   ├── topk/
│   ├── percentile/
│   └── tree/
│
└── papers/                         # Paper materials and supplementary data
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

TODO: Add license information.
