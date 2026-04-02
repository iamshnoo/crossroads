# Crossroads of Continents: Automated Artifact Extraction for Cultural Adaptation with Large Multimodal Models

Code and published assets for reproducing the Crossroads project.

DALLE Street dataset: https://huggingface.co/datasets/iamshnoo/dallestreet

This project is reproducible from:

- the GitHub repository
- the GitHub release assets under `data-assets-2026-03-30`
- your own GPU access and API credentials

The repo alone is not self-contained. The release archives provide the data trees and intermediate outputs that the scripts expect.

## 1. Environment Setup

Use Python 3.10 or 3.11.

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install \
  pandas numpy pillow requests tqdm openai datasets \
  matplotlib seaborn scikit-learn transformers accelerate \
  huggingface-hub opencv-python diffusers torch torchvision \
  torchmetrics supervision
```

If you want to run `edit.py`, also install GroundingDINO:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e ./GroundingDINO
```

Copy the example environment file:

```bash
cp env.example.sh env.sh
source env.sh
mkdir -p "$CROSSROADS_DOLLARSTREET_CACHE" "$CROSSROADS_MODEL_CACHE" "$CROSSROADS_INSTRUCTBLIP_CACHE" "$CROSSROADS_EDITS_CACHE"
```

Create `secrets.json` and point `CROSSROADS_SECRETS_FILE` to it if it is not in the repo root:

```json
{
  "AZURE_OPENAI_ENDPOINT": "...",
  "AZURE_OPENAI_API_KEY": "...",
  "GPT4V_OPENAI_ENDPOINT": "...",
  "GPT4V_OPENAI_API_KEY": "...",
  "OPENAI_API_KEY": "..."
}
```

Path variables used by the repo:

- `CROSSROADS_DATA_ROOT`: extracted project-data root; defaults to the repo root
- `CROSSROADS_SECRETS_FILE`: path to `secrets.json`
- `CROSSROADS_DOLLARSTREET_CACHE`: Hugging Face cache for Dollar Street
- `CROSSROADS_MODEL_CACHE`: cache root for LLaVA and related models
- `CROSSROADS_INSTRUCTBLIP_CACHE`: cache root for InstructBLIP
- `CROSSROADS_EDITS_CACHE`: cache root for editing and similarity models

Download the published release assets:

- Release: `data-assets-2026-03-30`
- URL: `https://github.com/iamshnoo/crossroads/releases/tag/data-assets-2026-03-30`

Reconstruct the split archives:

```bash
cat backup.tar.gz.part* > backup.tar.gz
cat dataset_dollarstreet.tar.gz.part* > dataset_dollarstreet.tar.gz
```

Verify them:

```bash
sha256sum backup.tar.gz
sha256sum dataset_dollarstreet.tar.gz
```

Expected SHA-256 values:

- `backup.tar.gz`: `94ba8c7e4c5413c5db31e6b8fa88c648d6bc0fe1e82ab2642398dda719396cff`
- `dataset_dollarstreet.tar.gz`: `cff780f4418bd40ae60aa7ed3a7fa3619d25de30e2178c94f8e63b9b1c92a97f`

Extract the archives. If you extract them outside the repo root, set:

```bash
export CROSSROADS_DATA_ROOT="/path/to/extracted/data/root"
```

## 2. Relevant Folders From Backup For Data And Intermediate Outputs

The release archives supply the folders below. These are the important ones for reproduction.

`results/`

- raw model outputs and generated artifacts
- examples used directly by scripts:
  `results/dalle_images.csv`
  `results/gpt/azure/*.csv`
  `results/llava/*.csv`
  `results/dalle_eval/.../*.csv`
  `results/marvl/.../*.csv`
  `results/dalle_objects/...`
  `results/human-study/study1/*.csv`
  `results/human-study/study2/*.csv`

`corrected/`

- normalized and aggregated outputs used for benchmarking and plotting
- this is the main comparison surface for reproducing the paper numbers

`marvl/`

- MARVL metadata and file mappings
- most importantly:
  `marvl/marvl_images.csv`
  plus the referenced MARVL image paths

These folders should exist under `CROSSROADS_DATA_ROOT`. If you extract the archives into the repo root, no extra path editing is required.

## 3. Code Files To Run For Each Step

Run the project in this order.

### Step A. Generate DALLE Street images

Primary script:

- `gen_dalle_street.py`

Command:

```bash
./.venv/bin/python gen_dalle_street.py --category car
```

This produces images under `results/dalle_natural/<category>/<country>/`. The script currently loops over all categories internally.

### Step B. Prepare or restore the DALLE image manifest

Required file:

- `results/dalle_images.csv`

This manifest is consumed by the DALLE Street evaluation and artifact-extraction scripts.

### Step C. Run geographic classification

Dollar Street:

- `classify_dollar_street_gpt.py`
- `classify_dollar_street_llava.py`

```bash
for split in car cups_mugs_glasses family_snapshots front_door home kitchen plate_of_food social_drink wall_decoration wardrobe; do
  ./.venv/bin/python classify_dollar_street_gpt.py --split "$split"
  ./.venv/bin/python classify_dollar_street_llava.py --split "$split"
done
```

DALLE Street:

- `classify_dalle_street_gpt.py`
- `classify_dalle_street_llava.py`

```bash
for kind in natural vivid; do
  for concept in car cups_mugs_glasses family_snapshots front_door home kitchen plate_of_food social_drink wall_decoration wardrobe; do
    ./.venv/bin/python classify_dalle_street_gpt.py --type "$kind" --concept "$concept"
    ./.venv/bin/python classify_dalle_street_llava.py --type "$kind" --concept "$concept"
  done
done
```

MARVL:

- `classify_marvl_gpt.py`
- `classify_marvl_llava.py`

```bash
for lang in id sw ta tr zh; do
  ./.venv/bin/python classify_marvl_gpt.py --country "$lang"
  ./.venv/bin/python classify_marvl_llava.py --country "$lang"
done
```

### Step D. Normalize raw outputs into benchmark-ready tables

- `classify_evals.py`
- `marvl_evals.py`

```bash
./.venv/bin/python classify_evals.py
./.venv/bin/python marvl_evals.py
```

### Step E. Produce aggregate metrics and paper plots

- `overall_acc.py`
- `plot.py`
- `plot_marvl.py`
- `data_stats.py`
- `human_evals.py`
- `rgb_calculate.py`
- `people_counter.py`

```bash
./.venv/bin/python overall_acc.py
./.venv/bin/python plot.py
./.venv/bin/python plot_marvl.py
./.venv/bin/python data_stats.py
./.venv/bin/python human_evals.py
./.venv/bin/python rgb_calculate.py
./.venv/bin/python people_counter.py
```

### Step F. Run the artifact-extraction pipeline

Run these in order:

1. `dalle_obj_detect.py`
2. `dalle_obj_process.py`
3. `dalle_obj_det_process.py`
4. `dalle_obj_counts.py`
5. `country_obj_dict_remove_adj.py`
6. `country_obj_cooccur.py`
7. `coocurrence_statistics.py`

### Step G. Optional image-editing and caption-editing workflow

Standalone helpers:

- `e2e-caption.py`
- `edit.py`

Recovered helper scripts:

- `evals/artifacts.py`
- `evals/cap-edit-captioning.py`
- `evals/cap-edit-caption-editing.py`
- `evals/cap-edit-image-editing-preprocess.py`
- `evals/cap-edit-pnp-config-maker.py`
- `evals/cap-edit-image-editing-pnp.py`
- `evals/cap-edit-eval.py`
- `evals/cultureadapt.py`
- `evals/eval_img_similarity.py`
- `evals/consolidate-eval-results.py`

## 4. Output Files To Compare Against For Benchmarking Against The Paper

These are the main reference files already present in the published release data. After rerunning the pipeline, compare your regenerated outputs against these files.

### Normalized benchmark tables

- `corrected/dollar_street_gpt.csv`
- `corrected/dollar_street_llava.csv`
- `corrected/dalle_street_gpt.csv`
- `corrected/dalle_street_llava.csv`
- `corrected/marvl_gpt.csv`
- `corrected/marvl_llava.csv`

### Aggregate benchmark CSVs

- `corrected/accuracy_by_income_quartiles_dollar_street_gpt.csv`
- `corrected/accuracy_by_income_quartiles_dollar_street_llava.csv`
- `corrected/country_wise_accuracy_by_subregion_dollar_street_gpt.csv`
- `corrected/country_wise_accuracy_by_subregion_dollar_street_llava.csv`
- `corrected/country_wise_accuracy_by_subregion_dalle_street_gpt.csv`
- `corrected/country_wise_accuracy_by_subregion_dalle_street_llava.csv`
- `corrected/country_wise_accuracy_dollar_street_gpt.csv`
- `corrected/country_wise_accuracy_dollar_street_llava.csv`
- `corrected/country_wise_accuracy_dalle_street_gpt.csv`
- `corrected/country_wise_accuracy_dalle_street_llava.csv`
- `corrected/marvl_accuracy_gpt.csv`
- `corrected/marvl_accuracy_llava.csv`

### Artifact-analysis comparison files

- `corrected/objects_proc.csv`
- `corrected/objects_proc_filtered.csv`
- `corrected/country_obj_dict_adj.json`
- `corrected/country_obj_dict_adj_unfiltered.json`
- `corrected/country_obj_dict_no_adj.json`
- `corrected/country_obj_dict_no_adj_unfiltered.json`
- `corrected/country_obj_adj_tfidf.csv`
- `corrected/country_obj_adj_tfidf.json`
- `corrected/country_obj_no_adj_tfidf.csv`
- `corrected/country_obj_no_adj_tfidf.json`

For paper reproduction, the most important comparison surface is still `corrected/`.

## 5. Results Files With Final Metrics

There is not a single `final_metrics.csv` in this repo. The final paper metrics are distributed across the benchmark CSVs below.

### Top-line model outputs used by `overall_acc.py`

- `corrected/dollar_street_llava.csv`
- `corrected/dollar_street_gpt.csv`
- `corrected/dalle_street_gpt.csv`
- `corrected/dalle_street_llava.csv`
- `corrected/marvl_gpt.csv`
- `corrected/marvl_llava.csv`

Running:

```bash
./.venv/bin/python overall_acc.py
```

prints the final headline accuracies from those six files.

### Final aggregate metric files

- `corrected/accuracy_by_income_quartiles_dollar_street_gpt.csv`
- `corrected/accuracy_by_income_quartiles_dollar_street_llava.csv`
- `corrected/country_wise_accuracy_by_subregion_dollar_street_gpt.csv`
- `corrected/country_wise_accuracy_by_subregion_dollar_street_llava.csv`
- `corrected/country_wise_accuracy_by_subregion_dalle_street_gpt.csv`
- `corrected/country_wise_accuracy_by_subregion_dalle_street_llava.csv`
- `corrected/country_wise_accuracy_dollar_street_gpt.csv`
- `corrected/country_wise_accuracy_dollar_street_llava.csv`
- `corrected/country_wise_accuracy_dalle_street_gpt.csv`
- `corrected/country_wise_accuracy_dalle_street_llava.csv`
- `corrected/marvl_accuracy_gpt.csv`
- `corrected/marvl_accuracy_llava.csv`

### Human-study and editing metrics

- `results/human-study/artifacts.csv`
- `results/consolidated_metrics.csv`
- `results/cap_edit/metrics/*.csv`
- `results/cultureadapt/metrics/*.csv`

If you can regenerate these files and they match the published release artifacts closely, you have reproduced the paper’s reported outputs.
