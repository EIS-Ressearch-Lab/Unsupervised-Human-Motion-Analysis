# Unsupervised-Human-Motion-Analysis

This repository contains a small reproducible demo for unsupervised human motion
analysis from wearable motion sensor signals. It is intentionally scoped to the
core analysis path:

- aligned multi-sensor motion snippets
- sliding-window feature extraction
- shared 2D embedding
- shared KDE atlas segmentation
- subject/label/region occupancy summaries
- a few representative final figures

The demo does not implement the full private raw-data import workflow. The
included `Data/` files are short, anonymised excerpts from already-aligned
sensor outputs.

The goal is that a reviewer can download the repository, install Python, and run
the demo without access to the full private dataset.

## Requirements

- Python 3.10 or newer
- Internet access the first time if Python packages are not already installed

On Windows, install Python from python.org and tick **Add python.exe to PATH**.
Alternatively, create a local `.venv` in this folder or set `DEMO_PYTHON` to a
specific `python.exe` path before running `run.cmd`.

The demo runner checks the required Python packages and runs:

```cmd
python -m pip install -r requirements.txt
```

if any dependency is missing.

## Run

From this folder:

```cmd
run.cmd
```

Run one subject only:

```cmd
run.cmd S1
```

Run a subset:

```cmd
run.cmd S1 S13
```

Outputs are written under:

```text
Results/
```

For a cohort run, the main output folder is:

```text
Results/cohort_shared/
```

For a single-subject run, the output folder is:

```text
Results/S1/
```

## Representative Outputs

The most useful files to inspect are:

- `step3_shared_embedding_all.csv`
- `step3_shared_embedding_by_group.png`
- `step3_shared_embedding_by_label.png`
- `step4_shared_atlas_heatmap.png`
- `step4_shared_atlas_overlay.png`
- `step4_shared_window_regions.csv`
- `step5_subject_label_region_occupancy.csv`
- `step7_region_activity_circular_network.png`
- `step7b_group_region_barplots_with_deltas.png`

## Repository Layout

```text
Data/                  small included demo snippets
demo_pipeline.py       minimal reproducible analysis pipeline
prepare_demo_data.py   developer utility for regenerating demo snippets
requirements.txt       Python dependencies
run.cmd                Windows one-click entry point
Results/               generated outputs, ignored by git
```

## Demo Data

The prepared demo dataset contains four short snippets:

- `S1`, `S2`: Control
- `S13`, `S17`: Injured

Each file contains:

- `Time`
- sparse `Marker` activity annotations
- `Group`
- all numeric sensor channels included in the demo snippets

The demo pipeline automatically detects all numeric sensor columns in each demo
file and uses them for sliding-window feature extraction.

The `Data/` directory is part of the demo release. If it is missing, re-download
the full repository.

`prepare_demo_data.py` is a developer utility used to regenerate the included
demo snippets from the full local project outputs. Users do not need it for a
normal demo run.

## Notes

This is a compact demonstration version of the analysis. It preserves the main
visual language and representative outputs of the full pipeline, but omits the
larger raw-data import and full-cohort processing steps.
