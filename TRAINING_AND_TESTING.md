# Training And Testing Reference

This is a reference workflow for training a new model and checking it on labeled data. Run commands from the repository root, the folder that contains `pyproject.toml` and `src/`.

## Training Input

Training uses the same feature-generation and graph-building pipeline as prediction, but the input table must include `ig_type`.

Required columns:

- `pdbid_chain`
- `igdomain_res_range`
- `ig_type`

Optional columns such as `score`, `seqid`, and `refpdbname` can stay in the table.

Stable labels:

```text
IgV
IgC1
IgC2
IgI
Cadherin
IgFN3
Lamin
CD19
```

Rows with labels outside this list are skipped during labeled graph generation and training.

Training requires enough labeled examples for stratified splitting. The code makes a held-out test split first, then uses 5-fold cross-validation on the remaining training/validation graphs. If any class has too few examples after the held-out split, training will stop with a clear error.

## Build Training Graphs

Use a labeled input file such as:

```text
input/train_domains.xlsx
```

Create structure, sequence, ESM, and graph files with the same steps used for prediction. Use training-specific output names so they do not overwrite prediction files:

```bash
gene-ig-identify features icn3d \
  --input-table input/train_domains.xlsx \
  --input-dir input \
  --node-executable node

gene-ig-identify features structures \
  --input-table input/train_domains.xlsx \
  --input-dir input \
  --pdb-subdir pdb_files \
  --structure-subdir structure_features_residues \
  --cutoff-distance 8

python src/create_sequences.py \
  --input-table input/train_domains.xlsx \
  --sequence-dir input/sequence_file \
  --output-file output/train_sequences.pkl.gz

mkdir -p /data/$USER/gene-ig-identify-esm-cache
gene-ig-create-esm-embeddings \
  --input-file output/train_sequences.pkl.gz \
  --output-file output/train_esm_embeddings.h5 \
  --model-name esm2_t33_650M_UR50D \
  --cache-dir /data/$USER/gene-ig-identify-esm-cache

python src/create_graphs.py \
  --input-table input/train_domains.xlsx \
  --pdb-dir input/pdb_files \
  --icn3dss-dir input/icn3dss \
  --structure-features-dir input/structure_features_residues \
  --icn3d-interactions-dir input/icn3d_interactions \
  --embeddings-file output/train_esm_embeddings.h5 \
  --graphs-output results/graphs/train_graphs.pt \
  --graph-lookup-output results/graphs/train_graph_lookup.pt
```

## Train Model

Default training output goes to `results/models`, because `config/default.yaml` sets:

```yaml
models_dir: results/models
```

Run:

```bash
gene-ig-identify train \
  --graphs-file results/graphs/train_graphs.pt \
  --graph-lookup-file results/graphs/train_graph_lookup.pt \
  --epochs 100 \
  --trials 30
```

To write model artifacts somewhere else:

```bash
gene-ig-identify train \
  --graphs-file results/graphs/train_graphs.pt \
  --graph-lookup-file results/graphs/train_graph_lookup.pt \
  --output-dir results/models \
  --epochs 100 \
  --trials 30
```

Training performs a held-out test split and stratified 5-fold cross-validation for Optuna hyperparameter tuning.

Expected training outputs:

```text
results/models/best_graph_model.pth
results/models/best_hyperparameters.json
results/models/model_config.json
results/models/cross_validation_summary.json
results/models/test_graphs.pt
results/models/test_labels.pt
results/models/loss_accuracy_plot_hybrid.png
```

The prediction workflow needs these three reusable files:

```text
results/models/best_graph_model.pth
results/models/best_hyperparameters.json
results/models/model_config.json
```

## Biowulf Training Notes

Run long training and ESM embedding jobs on an interactive or batch compute job, not the login node.

Example interactive GPU session:

```bash
sinteractive --gres=gpu:a100:1 --cpus-per-task=8 --mem=64g --time=12:00:00
module load python/3.11
source /data/$USER/gene-ig-identify-venv/bin/activate
cd /path/to/gene-ig-identify
```

Check CUDA visibility:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If `torch.cuda.is_available()` prints `False`, the job may still run on CPU, but training and ESM embeddings will be much slower.

## Testing A Trained Model On Labeled Data

The current public CLI uses `predict dataset` for model testing on any labeled or unlabeled dataset. For a labeled test file, keep the `ig_type` column in the Excel/CSV/TSV input. The prediction output will preserve the original rows and add:

```text
predicted_label
predicted_class_id
prediction_confidence
```

Build graph files for the test dataset:

```bash
gene-ig-identify features icn3d \
  --input-table input/test_domains.xlsx \
  --input-dir input \
  --node-executable node

gene-ig-identify features structures \
  --input-table input/test_domains.xlsx \
  --input-dir input \
  --pdb-subdir pdb_files \
  --structure-subdir structure_features_residues \
  --cutoff-distance 8

python src/create_sequences.py \
  --input-table input/test_domains.xlsx \
  --sequence-dir input/sequence_file \
  --output-file output/test_sequences.pkl.gz

gene-ig-create-esm-embeddings \
  --input-file output/test_sequences.pkl.gz \
  --output-file output/test_esm_embeddings.h5 \
  --model-name esm2_t33_650M_UR50D \
  --cache-dir /data/$USER/gene-ig-identify-esm-cache

python src/create_graphs.py \
  --input-table input/test_domains.xlsx \
  --pdb-dir input/pdb_files \
  --icn3dss-dir input/icn3dss \
  --structure-features-dir input/structure_features_residues \
  --icn3d-interactions-dir input/icn3d_interactions \
  --embeddings-file output/test_esm_embeddings.h5 \
  --graphs-output results/graphs/test_graphs.pt \
  --graph-lookup-output results/graphs/test_graph_lookup.pt
```

Run prediction with the trained model:

```bash
gene-ig-identify predict dataset \
  --graphs-file results/graphs/test_graphs.pt \
  --excel-file input/test_domains.xlsx \
  --model-dir results/models \
  --output-dir output
```

Expected outputs:

```text
output/test_domains_with_predictions.xlsx
output/test_domains_prediction_details.xlsx
```

Quick labeled accuracy check:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_excel("output/test_domains_with_predictions.xlsx")
df = df[df["ig_type"].notna()].copy()
accuracy = (df["ig_type"].astype(str) == df["predicted_label"].astype(str)).mean()

print(f"Rows evaluated: {len(df)}")
print(f"Accuracy: {accuracy:.4f}")
print(pd.crosstab(df["ig_type"], df["predicted_label"], rownames=["true"], colnames=["predicted"]))
PY
```

Rows predicted as `Other` are usually low-confidence predictions below the configured confidence threshold.
