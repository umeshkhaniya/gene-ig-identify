# Training And Testing

These instructions describe how to train and evaluate experiment models from an
already-built labeled graph dataset.

Run all commands from the repository root, the folder that contains
`pyproject.toml`, `src/`, and your input tables. Relative paths such as
`results/graphs/training_testing_graphs.pt` are resolved from this folder.

The existing EXP00 model in `results/models/` is the official 8-class baseline.
It is already trained and validated. Do not retrain it unless you explicitly
intend to replace or compare against a new EXP00 run. The organized copy of that
baseline is available at `results/experiments/EXP00/models/`.

For complete experiment definitions, see
[`docs/experiments.md`](docs/experiments.md).

## Files Used

You should have these files available:

```text
input_data.xlsx
results/graphs/training_testing_graphs.pt
results/graphs/training_testing_graph_lookup.pt
```

Use `input_data.xlsx`, not `input_data.xlxs`. The code supports `.xlsx`, `.xls`, `.csv`, `.tsv`, and `.txt`.

The `.pt` files are the actual training inputs:

- `results/graphs/training_testing_graphs.pt`: list of graph objects.
- `results/graphs/training_testing_graph_lookup.pt`: lookup of graph names to graph objects. The training workflow uses this lookup for splitting and training.
- `input_data.xlsx`: original row table used later for prediction-style testing and row-preserving output.

Each graph used for training must include a label from the selected experiment
mapping. The graph files should be built using the same experiment config that
will be used for training.

EXP00 labels:

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

EXP01 uses the EXP00 labels plus `IgE`, `IgFN3-like`, and `SOD`. EXP02
uses the EXP00 labels with `CD19` removed.

Training needs enough examples for stratified splitting. In practice, each class present in the graph dataset should have at least 6 examples, because the code first creates a held-out test split and then runs 5-fold cross-validation on the remaining training/evaluation graphs.

## Environment

Activate the environment where the package and dependencies are installed:

```bash
source gene-ig-identify-venv/bin/activate
```

If the package is not installed yet, install it from the repository root:

```bash
python -m pip install -e .
```

Confirm the CLI is available:

```bash
gene-ig-identify labels show
```

## Optional Smoke Test

Before a long run, you can check that the graph files load and the training
command starts correctly. Use a separate output directory for smoke tests so the
official EXP00 baseline is not overwritten:

```bash
gene-ig-identify --config config/experiments/exp00_8class.yaml train \
  --graphs-file results/graphs/training_testing_graphs.pt \
  --graph-lookup-file results/graphs/training_testing_graph_lookup.pt \
  --output-dir results/smoke/exp00_models_smoke_test \
  --epochs 2 \
  --trials 1
```

This is only a quick check. Do not use the smoke-test model as the final model.

## Training A New Experiment Model

Select the experiment with `--config`. When `--output-dir` is omitted, new
training runs write model artifacts and metrics into experiment-specific
directories:

```text
results/experiments/<EXPERIMENT_ID>/models/
results/experiments/<EXPERIMENT_ID>/metrics/
```

Example EXP01 training command:

```bash
gene-ig-identify --config config/experiments/exp01_12class.yaml train \
  --graphs-file results/graphs/exp01_training_graphs.pt \
  --graph-lookup-file results/graphs/exp01_training_graph_lookup.pt \
  --epochs 100 \
  --trials 30
```

Example EXP02 training command:

```bash
gene-ig-identify --config config/experiments/exp02_7class.yaml train \
  --graphs-file results/graphs/exp02_training_graphs.pt \
  --graph-lookup-file results/graphs/exp02_training_graph_lookup.pt \
  --epochs 100 \
  --trials 30
```

If you intentionally retrain EXP00 for a new comparison run, use either the
default experiment directory or an explicit comparison directory. Avoid using
`results/models/`, which is preserved as the legacy official baseline location.

```bash
gene-ig-identify --config config/experiments/exp00_8class.yaml train \
  --graphs-file results/graphs/training_testing_graphs.pt \
  --graph-lookup-file results/graphs/training_testing_graph_lookup.pt \
  --output-dir results/experiments/EXP00/models_retrained_comparison \
  --epochs 100 \
  --trials 30
```

For routine new experiment runs, prefer the experiment-specific defaults.

The training workflow does the following:

1. Loads labeled graphs from `results/graphs/training_testing_graph_lookup.pt`.
2. Creates a stratified held-out test split.
3. Runs Optuna hyperparameter tuning with stratified 5-fold cross-validation on the remaining graphs.
4. Trains a final model from scratch using the best hyperparameters.
5. Evaluates the final model on the held-out test split.

Expected model outputs for a new EXP01 run:

```text
results/experiments/EXP01/models/best_graph_model.pth
results/experiments/EXP01/models/best_hyperparameters.json
results/experiments/EXP01/models/model_config.json
results/experiments/EXP01/models/test_graphs.pt
results/experiments/EXP01/models/test_labels.pt
results/experiments/EXP01/metrics/cross_validation_summary.json
results/experiments/EXP01/metrics/loss_accuracy_plot_hybrid.png
```

EXP02 uses the same structure under `results/experiments/EXP02/`.

## Evaluation

Evaluation during training is handled by cross-validation and the final validation split. Inspect the saved summary files after training:

```bash
python -m json.tool results/experiments/EXP01/metrics/cross_validation_summary.json
python -m json.tool results/experiments/EXP01/models/model_config.json
```

Important fields:

- `best_mean_cv_accuracy`: mean accuracy from the best Optuna trial during 5-fold cross-validation.
- `best_value`: best cross-validation loss minimized by Optuna.
- `final_test_accuracy`: held-out test accuracy after final training.
- `test_accuracy`: same held-out test accuracy stored in `model_config.json`.
- `best_epoch`: final-training epoch whose checkpoint was saved.
- `best_validation_loss`: validation loss at the saved checkpoint.
- `best_validation_accuracy`: validation accuracy at the saved checkpoint.

Also review:

```text
results/experiments/EXP01/metrics/loss_accuracy_plot_hybrid.png
```

This plot shows training loss, validation loss, and validation accuracy across final training epochs.

## Testing On The Held-Out Split

The training command saves the held-out test graphs and labels:

```text
results/experiments/EXP01/models/test_graphs.pt
results/experiments/EXP01/models/test_labels.pt
```

The simplest held-out test result is already recorded in:

```text
results/experiments/EXP01/metrics/cross_validation_summary.json
results/experiments/EXP01/models/model_config.json
```

Use `final_test_accuracy` as the main test metric, because those graphs were not used for hyperparameter tuning or final model fitting.

To create a readable prediction table from an experiment held-out split, run:

```bash
python -m gene_ig_identify.scripts.evaluate_test_graphs \
  --model-dir results/experiments/EXP01/models \
  --test-graphs results/experiments/EXP01/models/test_graphs.pt \
  --test-labels results/experiments/EXP01/models/test_labels.pt
```

This writes:

```text
results/experiments/EXP01/predictions/heldout_test_predictions.csv
results/experiments/EXP01/predictions/heldout_test_predictions.xlsx
results/experiments/EXP01/metrics/heldout_classification_report.csv
results/experiments/EXP01/metrics/heldout_classification_report.xlsx
results/experiments/EXP01/metrics/heldout_confusion_matrix.csv
results/experiments/EXP01/metrics/heldout_confusion_matrix.xlsx
```

The classification report includes precision, recall, F1-score, support, macro average, and weighted average. Scores are rounded to two decimals, class support values are integers, and summary rows leave support blank.

## Testing With `input_data.xlsx`

You can also run the trained model across the full original table for a row-by-row prediction file:

```bash
gene-ig-identify --config config/experiments/exp00_8class.yaml predict dataset \
  --graphs-file results/graphs/training_testing_graphs.pt \
  --excel-file input_data.xlsx
```

Expected outputs:

```text
results/experiments/EXP00/predictions/input_data_with_predictions.xlsx
results/experiments/EXP00/predictions/input_data_prediction_details.xlsx
```

The same prediction dataset can be passed through EXP00, EXP01, and EXP02 once
each model exists. Each model interprets its outputs using the label mapping in
its own `model_config.json`.

This full-table test is useful for inspection, but it is not an independent held-out test if the same graphs were used for training. Treat the held-out `final_test_accuracy` from the training run as the cleaner testing metric.

## Quick Accuracy Check For Full-Table Predictions

After running `predict dataset`, calculate label agreement with the `ig_type` column:

```bash
python -m gene_ig_identify.scripts.evaluate_prediction_excel \
  --predictions-file results/experiments/EXP00/predictions/input_data_with_predictions.xlsx
```

Rows predicted as `Other` are low-confidence predictions below the configured confidence threshold.

`Other` is not a training class. The model output dimensions remain EXP00 = 8,
EXP01 = 11, and EXP02 = 7. A prediction below the existing `0.5` confidence
threshold is reported as `Other`, but `Other` is not included in any label
mapping or `model_config.json` label list.

Prediction does not use the true label to determine the model output. If the
input table includes `ig_type`, that value is preserved for later evaluation
only. For example, an EXP02 model can receive a row whose true label is CD19 and
still produce one of its seven model outputs.

## Notes

- Use a GPU for long training runs when possible. The code automatically uses CUDA if `torch.cuda.is_available()` is true.
- Increase `--trials` for a broader hyperparameter search.
- Increase `--epochs` for longer final training.
- If training fails with a stratification error, check class counts in the graph dataset and add more labeled examples for the smallest class.
- If `predict dataset` reports row alignment problems, make sure `results/graphs/training_testing_graphs.pt` was built from the same `input_data.xlsx` rows and label values.
- The experiment reorganization does not change the iCn3D feature pipeline or
  the iCn3D-derived features used by graph generation.
