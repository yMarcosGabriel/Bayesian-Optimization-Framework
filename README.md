# A Bayesian Optimization Framework for Joint Frequency - Permittivity Inverse Design of Terahertz-Band Micro-Scale Rectangular Dielectric Resonator Antennas

Adaptive Bayesian-optimization pipeline for the inverse design of a micro-scale
rectangular dielectric resonator antenna (DRA) using Ansys HFSS, Gaussian-process
surrogate modeling, Expected Improvement, feasibility classification, and
adaptive search-bound remodeling.

This repository accompanies the manuscript *"A Bayesian Optimization Framework for Joint Frequency - Permittivity Inverse Design of Terahertz-Band Micro-Scale Rectangular Dielectric Resonator Antennas"* (currently under submission - see [Citing this work](#citing-this-work)).

## What this pipeline does

1. **Parametric HFSS simulation** — evaluates candidate antenna geometries over
   target resonant frequencies and relative-permittivity values, extracting
   $S_{11}$, $Z_{11}$, VSWR, bandwidth, and resonant frequency.
2. **Latin Hypercube initialization** — generates the initial design of
   experiments over the geometric search space.
3. **Gaussian-process surrogate modeling** — learns the objective function from
   HFSS evaluations and uses Expected Improvement (EI) to select new candidate
   geometries.
4. **Feasibility-aware acquisition** — trains a Gaussian-process classifier to
   estimate the probability that a candidate will produce a valid electromagnetic
   response and weights EI accordingly.
5. **Adaptive search-bound remodeling** — detects persistent pressure at the
   current search-space boundaries and expands the affected geometric variables
   in a coordinated manner.
6. **Fabrication-oriented validation** — rounds the best geometry to the selected
   fabrication precision and revalidates it in HFSS before generating the final
   tables and publication-quality figures.

## Repository structure

```text
.
├── pipeline_bo.py                      # main Bayesian-optimization pipeline
├── Pipeline_BO_Comp.ipynb              # cleaned notebook version
├── REPRODUCIBILITY.md                  # execution and provenance notes
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment specification
├── .env.example                        # local HFSS path configuration example
├── .gitignore                           # excludes local/HFSS files
├── LICENSE                             # MIT License (code)
├── CITATION.cff                        # citation metadata
└── README.md
```

## Installation

```bash
git clone https://github.com/yMarcosGabriel/Bayesian-Optimization-Framework.git
cd Bayesian-Optimization-Framework
pip install -r requirements.txt
```

The optimization workflow requires a compatible Python environment together
with `numpy`, `pandas`, `matplotlib`, `scipy`, and `scikit-learn`.

The HFSS-dependent stages additionally require `pyaedt`, Ansys Electronics
Desktop/HFSS, and a valid Ansys license. The repository does **not** contain
the proprietary HFSS project file.

## Reproducing the optimization

The pipeline can be executed with a local HFSS project. Before running, define:

```bash
export HFSS_PROJECT_PATH=/path/to/your/local/project.aedt
export RESULTS_DIR=results
```

On Windows PowerShell:

```powershell
$env:HFSS_PROJECT_PATH="C:\path\to\your\project.aedt"
$env:RESULTS_DIR="results"
```

Then run:

```bash
python pipeline_bo.py
```

The target frequency and relative-permittivity grids are defined in the
configuration section of `pipeline_bo.py`.

The default workflow performs the complete frequency/$\varepsilon_r$ grid,
Bayesian optimization, adaptive bound remodeling, final rounded-design
validation, and figure generation.

## HFSS project and licensing

The original `.aedt` HFSS project is **not distributed** in this repository.

To reproduce the complete electromagnetic workflow, users must provide their
own compatible HFSS project and valid Ansys license. The external HFSS project
must be accessible through the `HFSS_PROJECT_PATH` environment variable.

## Generated outputs

When executed locally, the pipeline generates the following outputs for each
target `(frequency, relative permittivity)` pair:

| File | Description |
|---|---|
| `resultados_bo.csv` | Complete optimization history, including geometric variables, objective value, iteration, optimization phase, feasibility, and electromagnetic metrics. |
| `tabela_resultado_final.csv` | Final fabrication-rounded geometry and validated electromagnetic metrics. |
| `historico_remodelagem_bounds.csv` | Search-bound history and boundary-pressure diagnostics used by the adaptive strategy. |
| `historico_adaptativo.csv` | Summary of adaptive optimization decisions. |
| `resumo_grade_er_frequencia.csv` | Summary of the complete frequency/$\varepsilon_r$ optimization grid. |
| `fig1_convergencia.png` | Bayesian-optimization convergence plot. |
| `fig2_s11_final.png` | Final $S_{11}$ response. |
| `fig3_exploracao_espaco.png` | Parameter-space exploration plot. |
| `fig4_ei_decay.png` | Acquisition-function decay plot, when available. |

These files are generated locally during execution and are not distributed in
this repository because the underlying electromagnetic simulation data are
subject to the licensing restrictions described in [Data availability](#data-availability).

## Data availability

The source code and documentation are openly available in this repository. However, the datasets and electromagnetic simulation data generated through Ansys HFSS are **not distributed** because their redistribution is restricted by the applicable Ansys licensing terms.

This includes the HFSS project files, raw simulation outputs, and simulation-derived datasets generated during the optimization workflow.

The optimization pipeline is provided in full, allowing the methodology, algorithmic implementation, configuration, and data-processing procedures to be inspected and reused. Reproduction of the complete electromagnetic optimization, however, requires access to a compatible Ansys HFSS installation and the corresponding licensed simulation environment.

Accordingly, the public repository contains the open-source code and documentation, but does not include the restricted HFSS simulation data.

## Citing this work

This manuscript is currently under submission and does not yet have a
journal DOI. Until the journal article receives its final DOI, please cite
the software release using the metadata in [`CITATION.cff`](./CITATION.cff).

The canonical repository is:

[**Bayesian-Optimization-Framework**](https://github.com/yMarcosGabriel/Bayesian-Optimization-Framework)

## License

* Code and notebook: [MIT License](./LICENSE)
* Third-party software, including Ansys Electronics Desktop/HFSS and PyAEDT,
  remains subject to its respective licenses.
* HFSS project files and other restricted third-party assets are not covered
  by the MIT license.

## Contact

Marcos Gabriel Santos— Graduate Program in Electrical Engineering,
Universidade Federal do Pará (UFPA), Belém, PA, Brazil.
