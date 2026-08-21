# Experiments

This project currently defines three Ig-domain classification experiments. The
experiment label space is configuration driven: the selected YAML file provides
the experiment ID, experiment name, and ordered label list.

Use these files to select an experiment:

```text
config/experiments/exp00_8class.yaml
config/experiments/exp01_11class.yaml
config/experiments/exp02_7class.yaml
```

The default configuration is EXP00.

Pass `--config` before the subcommand to select a non-default experiment:

```bash
gene-ig-identify --config config/experiments/exp00_8class.yaml predict dataset ...
gene-ig-identify --config config/experiments/exp01_11class.yaml train ...
gene-ig-identify --config config/experiments/exp02_7class.yaml train ...
```

EXP01 and EXP02 prediction commands should be run only after those experiment
models have been trained and saved in their experiment model directories.

## EXP00: 8class_baseline

EXP00 is the existing 8-class baseline model and the reference experiment for
the project.

Class mapping:

```text
IgV      = 0
IgC1     = 1
IgC2     = 2
IgI      = 3
Cadherin = 4
IgFN3    = 5
Lamin    = 6
CD19     = 7
```

The trained EXP00 baseline is available in two places:

```text
results/models/
results/experiments/EXP00/models/
```

`results/models/` is preserved for backward compatibility with older commands
and documentation. `results/experiments/EXP00/` is the organized experiment
location used by the current repository layout.

## EXP01: 11class_expanded

EXP01 defines an 11-class expanded label space. It includes all EXP00 classes
plus:

```text
IgE
IgFN3-like
SOD
```

Class mapping:

```text
IgV        = 0
IgC1       = 1
IgC2       = 2
IgI        = 3
Cadherin   = 4
IgFN3      = 5
Lamin      = 6
CD19       = 7
IgE        = 8
IgFN3-like = 9
SOD        = 10
```

EXP01 evaluates the classification problem using an expanded label space with
additional classes. EXP01 is defined but has not yet been newly trained and
evaluated in this repository state. Do not interpret the existence of the
configuration as a performance result.

## EXP02: 7class_no_CD19

EXP02 defines a 7-class reduced label space. CD19 is intentionally removed from
the EXP00 label space.

Class mapping:

```text
IgV      = 0
IgC1     = 1
IgC2     = 2
IgI      = 3
Cadherin = 4
IgFN3    = 5
Lamin    = 6
```

EXP02 evaluates the classification problem using a reduced label space without
CD19. EXP02 is defined but has not yet been newly trained and evaluated in this
repository state. Do not interpret the existence of the configuration as a
performance result.

## Model Classes And Other

`Other` is not a trained class. It is a prediction-time label produced only by
the existing confidence-threshold rule.

The neural network output dimensions are:

```text
EXP00 = 8
EXP01 = 11
EXP02 = 7
```

The prediction workflow uses a confidence threshold of `0.5`:

```text
if maximum predicted probability >= 0.5:
    predicted_label = corresponding model-specific class label

if maximum predicted probability < 0.5:
    predicted_label = "Other"
```

Therefore `Other` is not included in:

```text
LABEL_MAPPING
REVERSE_LABEL_MAPPING
experiment labels
num_classes
model output dimensions
model_config.json label lists
```

Do not describe `Other` as an additional model class.

## Prediction Versus Evaluation

Prediction does not use the true label to determine the model output. If the
input table contains an `ig_type` column, that value is preserved for later
evaluation, but it does not alter the prediction.

For example, EXP02 has no CD19 training class. If an EXP02 model receives a
sample whose true label is CD19, the model still produces one of its seven
outputs:

```text
true_label = CD19
predicted_label = IgFN3
```

That is a valid model prediction. The fact that CD19 is outside the EXP02
training label space should be handled later during evaluation and comparison.
The prediction workflow does not add an unknown class, does not emit
`OUT_OF_LABEL_SPACE`, and does not reject the sample because of the true label.

## Same Prediction Dataset

The intended comparison design is to run the same prediction dataset through
each trained model:

```text
same prediction data
        |
        +----> EXP00 model -> 8-class prediction
        |
        +----> EXP01 model -> 11-class prediction
        |
        +----> EXP02 model -> 7-class prediction
```

Each model interprets its output using the label mapping saved with that model
in `model_config.json`. True labels remain evaluation data and are not used to
alter prediction.

The three models have not yet been compared in this repository state.

## Results Directory

The intended artifact layout is:

```text
results/
|-- models/
|   `-- legacy EXP00 artifacts
|
`-- experiments/
    |-- EXP00/
    |   |-- models/
    |   |-- metrics/
    |   `-- predictions/
    |
    |-- EXP01/
    |   |-- models/
    |   |-- metrics/
    |   `-- predictions/
    |
    `-- EXP02/
        |-- models/
        |-- metrics/
        `-- predictions/
```

`results/models/` contains the legacy EXP00 artifacts and is preserved for
backward compatibility.

`results/experiments/EXP00/` contains the organized copy of the existing EXP00
baseline artifacts.

Future EXP01 and EXP02 training outputs should be isolated in their respective
experiment directories.

## Model Metadata

New `model_config.json` files contain:

```text
experiment
name
labels
label_mapping
num_classes
```

This lets each trained model carry its own label space.

Conceptually:

```text
EXP01 model
    |
model_config.json
    |
11 labels
    |
prediction uses those 11 labels

EXP02 model
    |
model_config.json
    |
7 labels
    |
prediction uses those 7 labels
```

Older EXP00 artifacts may not contain the newer label metadata. Those artifacts
remain readable through the legacy EXP00 mapping.

## iCn3D Pipeline

The experiment reorganization does not change the iCn3D feature pipeline.

All existing iCn3D-related features remain part of the project. EXP00, EXP01,
and EXP02 use the same underlying feature generation pipeline unless an existing
configuration explicitly specifies otherwise.
