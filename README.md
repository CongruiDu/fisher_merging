# fisher_merge

1. compute one diagonal Fisher file per model
2. evaluate three baselines on a GLUE split: output ensemble, isotropic parameter merge, and Fisher merge

## Files

- `get_fisher.py`: computes a diagonal Fisher for one model, restricted to parameters that are shared across the full merge set
- `get_fisher.sh`: minimal RTE-only example that writes Fisher files to `fishers/`
- `merge_and_evaluate.py`: loads models and Fisher files, then reports `Ensemble`, `Isotropic merged`, and `Fisher merged`
- `run_glue_tasks.sh`: end-to-end wrapper for `rte`, `mrpc`, and `sst2`; it computes Fishers first and then runs `merge_and_evaluate.py`
- `task_model_registry.py`: paper-aligned checkpoint lists for `paper_exact` and `paper_available`
- `requirements.txt`: Python dependencies

## Environment

Create a conda environment and install the dependencies:

```bash
conda create -n fisher-merge python=3.10 -y
conda activate fisher-merge
cd path/to/fisher_merge
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Start

Run the default three-task pipeline:

```bash
bash run_glue_tasks.sh
```

This uses:
- tasks: `rte mrpc sst2`
- model variant: `paper_available`
- outputs under `runs/paper_available/<task>/`

Run only selected tasks:

```bash
bash run_glue_tasks.sh rte sst2
```