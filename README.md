# gene-ig-identify
Gene Ig can predict immunoglobulin (Ig) and Ig-like domains in protein structures. The main goal is to quantify Ig domains in the human genome at both the domain and chain levels. Ig domains are important building blocks in antibodies, T cell receptors, and many cell-surface receptors. Counting and classifying them can help us better understand immune diversity, gene-family expansion, and possible links to immune function and disease. This approach can be applied to any genome.
The graph is created using structural features and sequence features from ESMFold embeddings, and it uses a graph neural network to identify Ig-domain types more precisely than broad structural searches alone.

The active workflow lives under `src/`.

## Repository Layout

```text
docs/
  experiments.md                   # EXP00/EXP01/EXP02 definitions and artifact layout
src/
  create_graphs.py                 # graph-building script
  create_sequences.py              # per-domain sequence extraction script
  create_esm_embeddings.py         # ESM-2 embedding script
  gene_ig_identify/                # installable package code
    cli.py
    features/
    integrations/
      js/                          # iCn3D Node.js helper scripts
    io/
    models/
    scripts/
    workflows/
config/
  default.yaml                     # default paths and runtime settings
  experiments/                     # experiment-specific label spaces
results/
  models/                          # legacy EXP00 model artifacts
  experiments/
    EXP00/
      models/                      # organized EXP00 baseline model artifacts
      metrics/
      predictions/
    EXP01/
      models/
      metrics/
      predictions/
    EXP02/
      models/
      metrics/
      predictions/
```

## Install

Python 3.11 or newer is required. Check whether it is already installed:

```bash
python3.11 --version
```

If `python3.11` is not available, install it first. On macOS with Homebrew:

```bash
brew install python@3.11
```

With conda or mamba, replacing `conda` with `mamba` if needed:

```bash
conda create -n gene-ig-identify-python python=3.11
conda activate gene-ig-identify-python
```

Then clone and install the project. If you installed Python with Homebrew or another system package manager, create the project venv:

```bash
git clone https://github.com/umeshkhaniya/gene-ig-identify.git
cd gene-ig-identify
python3.11 -m venv gene-ig-identify-venv
source gene-ig-identify-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you are using the activated conda environment, install directly inside it:

```bash
git clone https://github.com/umeshkhaniya/gene-ig-identify.git
cd gene-ig-identify
python -m pip install --upgrade pip
python -m pip install -e .
```

On NIH Biowulf, use the Python module instead of installing Python yourself:

```bash
module -t avail python
module load python/3.11
python --version
git clone https://github.com/umeshkhaniya/gene-ig-identify.git
cd gene-ig-identify
python -m venv --system-site-packages /data/$USER/gene-ig-identify-venv
source /data/$USER/gene-ig-identify-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If `/data/$USER` is not available for your account, create the venv in another writable project or data directory. The `--system-site-packages` option lets the venv see packages from Biowulf's loaded Python module. For long ESM/model runs, use an interactive or batch compute job rather than running on the login node.

Important dependency notes:

- `fair-esm` is installed from [`pyproject.toml`](pyproject.toml); the first embedding run may download ESM-2 model weights into the PyTorch cache.
- iCn3D is a Node.js dependency. Install Node.js/npm and the iCn3D npm packages separately before running the iCn3D feature step. See the iCn3D npm package: <https://www.npmjs.com/package/icn3d>.

## Input Table

The input table can be Excel, CSV, or TSV. It must include `pdbid_chain` and `igdomain_res_range`.

```text
pdbid_chain	igdomain_res_range	score	seqid	refpdbname	ig_type
Q02413_A	55_148	0.867492	0.309278	ECadherin_4zt1A_human_n2	Cadherin
Q02413_A	164_260	0.802861	0.21978	ECadherin_4zt1A_human_n2	Cadherin
```

Required columns:

- `pdbid_chain`: structure ID and chain, such as `Q02413_A`
- `igdomain_res_range`: domain residue range, such as `55_148`

Use the exact column name `igdomain_res_range`.

For prediction, `ig_type` can be omitted because it is the value the model predicts. For training, include `ig_type`.

## Experiments

The project defines three experiment label spaces. The selected configuration
file controls the experiment ID, experiment name, and ordered label list.

```text
EXP00: 8class_baseline
  IgV, IgC1, IgC2, IgI, Cadherin, IgFN3, Lamin, CD19

EXP01: 11class_expanded
  EXP00 labels plus IgE, IgFN3-like, SOD

EXP02: 7class_no_CD19
  EXP00 labels with CD19 removed
```

The default configuration is EXP00. The existing trained EXP00 baseline is
available at `results/experiments/EXP00/models/`; the legacy
`results/models/` location is also preserved for backward compatibility.

Detailed label mappings, model metadata behavior, result directories, and
prediction/evaluation distinctions are documented in
[`docs/experiments.md`](docs/experiments.md).

## Predict New Ig Domains

Run the commands in this section from the repository root, the folder that contains `pyproject.toml` and `src/`. On Biowulf, this usually means running `cd gene-ig-identify` after cloning the repository.

Save new domains in a file such as:

```text
input/new_domains.xlsx
```

Create the standard folders:

```bash
mkdir -p input/pdb_files input/sequence_file input/icn3dss input/icn3d_interactions
mkdir -p input/structure_features_residues output results/graphs
```

Install the Node.js packages in the repo root:

```bash
npm install icn3d three jquery jsdom axios querystring
```

Generate iCn3D-derived files:

```bash
gene-ig-identify features icn3d \
  --input-table input/new_domains.xlsx \
  --input-dir input \
  --node-executable node
```

Generate residue structure features:

```bash
gene-ig-identify features structures \
  --input-table input/new_domains.xlsx \
  --input-dir input \
  --pdb-subdir pdb_files \
  --structure-subdir structure_features_residues \
  --cutoff-distance 8
```

Extract per-domain sequences:

```bash
python src/create_sequences.py \
  --input-table input/new_domains.xlsx \
  --sequence-dir input/sequence_file \
  --output-file output/new_domain_sequences.pkl.gz
```

Create ESM-2 embeddings:

```bash
mkdir -p /data/$USER/gene-ig-identify-esm-cache
python src/create_esm_embeddings.py \
  --input-file output/new_domain_sequences.pkl.gz \
  --output-file output/new_domain_esm_embeddings.h5 \
  --model-name esm2_t33_650M_UR50D \
  --cache-dir /data/$USER/gene-ig-identify-esm-cache
```

After installation, you can also run the installed command instead of referencing `src/create_esm_embeddings.py` directly:

```bash
mkdir -p /data/$USER/gene-ig-identify-esm-cache
gene-ig-create-esm-embeddings \
  --input-file output/new_domain_sequences.pkl.gz \
  --output-file output/new_domain_esm_embeddings.h5 \
  --model-name esm2_t33_650M_UR50D \
  --cache-dir /data/$USER/gene-ig-identify-esm-cache
```

On Biowulf, keep this cache in `/data/$USER` so the large ESM-2 model download does not fill `/home/$USER/.cache`.

Build unlabeled prediction graphs:

```bash
python src/create_graphs.py \
  --input-table input/new_domains.xlsx \
  --pdb-dir input/pdb_files \
  --icn3dss-dir input/icn3dss \
  --structure-features-dir input/structure_features_residues \
  --icn3d-interactions-dir input/icn3d_interactions \
  --embeddings-file output/new_domain_esm_embeddings.h5 \
  --graphs-output results/graphs/new_domain_graphs.pt \
  --graph-lookup-output results/graphs/new_domain_graph_lookup.pt
```

Predict labels with the saved model:

```bash
gene-ig-identify --config config/experiments/exp00_8class.yaml predict dataset \
  --graphs-file results/graphs/new_domain_graphs.pt \
  --excel-file input/new_domains.xlsx
```

Prediction outputs:

```text
results/experiments/EXP00/predictions/new_domains_with_predictions.xlsx
results/experiments/EXP00/predictions/new_domains_prediction_details.xlsx
```

The model directory must contain:

```text
results/experiments/EXP00/models/best_graph_model.pth
results/experiments/EXP00/models/best_hyperparameters.json
results/experiments/EXP00/models/model_config.json
```

To use the preserved legacy baseline location explicitly, pass
`--model-dir results/models`. Prediction behavior is the same because older
EXP00 artifacts remain readable.

The prediction workflow uses the label mapping saved with the loaded model.
For low-confidence predictions, the existing threshold rule is unchanged:
probability below `0.5` is reported as `Other`. `Other` is not a trained class.

## Expected Feature Files

The graph step expects these files to exist for each row. For the structure file, use either `.pdb` or `.cif`.

```text
input/pdb_files/<PDB>.pdb  or  input/pdb_files/<PDB>.cif
input/sequence_file/<PDB>_sequence.pkl.gz
input/icn3dss/<PDB>_icn3dss.pkl.gz
input/icn3d_interactions/<PDB>_<CHAIN>_icn3dinteraction.json
input/structure_features_residues/<PDB>_<CHAIN>_<BEGIN>_<END>_structure.pkl.gz
```

For `Q02413_A` with `igdomain_res_range` `55_148`, expected names include either `Q02413.pdb` or `Q02413.cif`:

```text
input/pdb_files/Q02413.pdb  or  input/pdb_files/Q02413.cif
input/sequence_file/Q02413_sequence.pkl.gz
input/icn3dss/Q02413_icn3dss.pkl.gz
input/icn3d_interactions/Q02413_A_icn3dinteraction.json
input/structure_features_residues/Q02413_A_55_148_structure.pkl.gz
```

## Other Commands

The full package CLI is available after installation:

```bash
gene-ig-identify labels show
gene-ig-identify config validate
gene-ig-identify features icn3d --input-table input/new_domains.xlsx --input-dir input
gene-ig-identify features structures --input-table input/new_domains.xlsx --input-dir input
gene-ig-identify sequences extract --input-table input/new_domains.xlsx --output-file output/new_domain_sequences.pkl.gz
gene-ig-identify embeddings esm --input-file output/new_domain_sequences.pkl.gz --output-file output/new_domain_esm_embeddings.h5 --cache-dir /data/$USER/gene-ig-identify-esm-cache
gene-ig-identify graphs build --input-table input/new_domains.xlsx --embeddings-file output/new_domain_esm_embeddings.h5 --graphs-output results/graphs/new_domain_graphs.pt --graph-lookup-output results/graphs/new_domain_graph_lookup.pt
gene-ig-identify --config config/experiments/exp00_8class.yaml predict dataset --graphs-file results/graphs/new_domain_graphs.pt --excel-file input/new_domains.xlsx
python -m gene_ig_identify.scripts.evaluate_test_graphs --help
python -m gene_ig_identify.scripts.evaluate_prediction_excel --predictions-file results/experiments/EXP00/predictions/input_data_with_predictions.xlsx
```

Use `--help` on any command to see all arguments.

For model training and labeled testing/evaluation notes, see [`TRAINING_AND_TESTING.md`](TRAINING_AND_TESTING.md).

## Labels

Label spaces are configuration driven. EXP00 remains the default 8-class
baseline and is still exposed through `LABEL_MAPPING` and
`REVERSE_LABEL_MAPPING` for backward compatibility. EXP01 and EXP02 use their
own ordered label lists from `config/experiments/`.

Rows with labels outside the selected experiment mapping are skipped during
labeled graph generation. Training validates that graph labels fit the selected
experiment label space.

`Other` is not part of any model label mapping. It is only a prediction-time
output used when the maximum class probability is below the existing `0.5`
confidence threshold.

## Configuration

Default paths live in `config/default.yaml`.

Experiment configs live in:

```text
config/experiments/exp00_8class.yaml
config/experiments/exp01_11class.yaml
config/experiments/exp02_7class.yaml
```

You can override them with:

- `--config path/to/config.yaml`
- environment variables such as `GENE_IG_IDENTIFY_INPUT_DIR`
- direct command-line arguments like `--pdb-dir` or `--embeddings-file`
