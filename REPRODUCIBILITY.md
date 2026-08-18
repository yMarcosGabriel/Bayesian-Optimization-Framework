# Reproducibility notes

## Computational workflow

The pipeline:

1. creates an initial Latin Hypercube sample;
2. evaluates candidate geometries in HFSS;
3. extracts S11 and Z11 information;
4. computes a combined objective from resonance frequency, S11, impedance and VSWR;
5. trains a Gaussian-process surrogate;
6. uses Expected Improvement weighted by a feasibility classifier;
7. performs adaptive search-bound remodeling when persistent boundary pressure is detected;
8. validates the fabrication-rounded optimum;
9. produces tabular outputs and figures.

## External dependency

HFSS/AEDT is an external licensed dependency. The repository deliberately does not contain the `.aedt` project.

## Environment variables

`HFSS_PROJECT_PATH`
: Absolute or relative path to the local HFSS `.aedt` project.

`RESULTS_DIR`
: Output directory. Defaults to `results`.

## Reproducibility limitation

Without the licensed HFSS model and compatible Ansys environment, the complete electromagnetic optimization cannot be rerun from the repository alone. The public repository is therefore intended to preserve the optimization algorithm, configuration, provenance and any legally shareable derived results rather than the proprietary simulation environment.
