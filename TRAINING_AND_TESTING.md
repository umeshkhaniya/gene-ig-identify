# Training And Testing

These instructions retrain the graph model from scratch using an already-built labeled graph dataset.

Run all commands from the repository root, the folder that contains `pyproject.toml`, `src/`, and `input_data.xlsx`. Relative paths such as `results/graphs/training_testing_graphs.pt` are resolved from this folder.

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

Each graph used for training must include a label from the stable mapping:

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

Before a long run, you can check that the graph files load and the training command starts correctly:

```bash
gene-ig-identify train \
  --graphs-file results/graphs/training_testing_graphs.pt \
  --graph-lookup-file results/graphs/training_testing_graph_lookup.pt \
  --output-dir results/models_smoke_test \
  --epochs 2 \
  --trials 1
```

This is only a quick check. Do not use the smoke-test model as the final model.

## Training From Scratch

Run training into `results/models`. If files already exist there, retraining updates them with the new model artifacts:

```bash
gene-ig-identify train \
  --graphs-file results/graphs/training_testing_graphs.pt \
  --graph-lookup-file results/graphs/training_testing_graph_lookup.pt \
  --output-dir results/models \
  --epochs 100 \
  --trials 30
```

The training workflow does the following:

1. Loads labeled graphs from `results/graphs/training_testing_graph_lookup.pt`.
2. Creates a stratified held-out test split.
3. Runs Optuna hyperparameter tuning with stratified 5-fold cross-validation on the remaining graphs.
4. Trains a final model from scratch using the best hyperparameters.
5. Evaluates the final model on the held-out test split.

Expected model outputs:

```text
results/models/best_graph_model.pth
results/models/best_hyperparameters.json
results/models/model_config.json
results/models/cross_validation_summary.json
results/models/test_graphs.pt
results/models/test_labels.pt
results/models/loss_accuracy_plot_hybrid.png
```

## Evaluation

Evaluation during training is handled by cross-validation and the final validation split. Inspect the saved summary files after training:

```bash
python -m json.tool results/models/cross_validation_summary.json
python -m json.tool results/models/model_config.json
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
results/models/loss_accuracy_plot_hybrid.png
```

This plot shows training loss, validation loss, and validation accuracy across final training epochs.

## Testing On The Held-Out Split

The training command saves the held-out test graphs and labels:

```text
results/models/test_graphs.pt
results/models/test_labels.pt
```

The simplest held-out test result is already recorded in:

```text
results/models/cross_validation_summary.json
results/models/model_config.json
```

Use `final_test_accuracy` as the main test metric, because those graphs were not used for hyperparameter tuning or final model fitting.

To create a readable prediction table from `results/models/test_graphs.pt`, run:

```bash
python -m gene_ig_identify.scripts.evaluate_test_graphs \
  --model-dir results/models \
  --test-graphs results/models/test_graphs.pt \
  --test-labels results/models/test_labels.pt
```

This writes:

```text
results/models/heldout_test_predictions.csv
results/models/heldout_test_predictions.xlsx
```

## Testing With `input_data.xlsx`

You can also run the trained model across the full original table for a row-by-row prediction file:

```bash
gene-ig-identify predict dataset \
  --graphs-file results/graphs/training_testing_graphs.pt \
  --excel-file input_data.xlsx \
  --model-dir results/models \
  --output-dir output
```

Expected outputs:

```text
output/input_data_with_predictions.xlsx
output/input_data_prediction_details.xlsx
```

This full-table test is useful for inspection, but it is not an independent held-out test if the same graphs were used for training. Treat the held-out `final_test_accuracy` from the training run as the cleaner testing metric.

## Quick Accuracy Check For Full-Table Predictions

After running `predict dataset`, calculate label agreement with the `ig_type` column:

```bash
python -m gene_ig_identify.scripts.evaluate_prediction_excel \
  --predictions-file output/input_data_with_predictions.xlsx
```

Rows predicted as `Other` are low-confidence predictions below the configured confidence threshold.

## Notes

- Use a GPU for long training runs when possible. The code automatically uses CUDA if `torch.cuda.is_available()` is true.
- Increase `--trials` for a broader hyperparameter search.
- Increase `--epochs` for longer final training.
- If training fails with a stratification error, check class counts in the graph dataset and add more labeled examples for the smallest class.
- If `predict dataset` reports row alignment problems, make sure `results/graphs/training_testing_graphs.pt` was built from the same `input_data.xlsx` rows and label values.
