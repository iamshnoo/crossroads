# Crossroads of Continents: Automated Artifact Extraction for Cultural Adaptation with Large Multimodal Models

Code and assets for the Crossroads project.

DALLE Street dataset: https://huggingface.co/datasets/iamshnoo/dallestreet

## Data Availability

The local project archives used for this work are now published as GitHub release assets:

- Release: `data-assets-2026-03-30`
- URL: `https://github.com/iamshnoo/crossroads/releases/tag/data-assets-2026-03-30`

Because GitHub release assets must be smaller than 2 GiB each, the archives are split into numbered parts:

- `backup.tar.gz.part001` ... `backup.tar.gz.part015`
- `dataset_dollarstreet.tar.gz.part001` ... `dataset_dollarstreet.tar.gz.part051`

After downloading all parts, reconstruct them locally with:

```bash
cat backup.tar.gz.part* > backup.tar.gz
cat dataset_dollarstreet.tar.gz.part* > dataset_dollarstreet.tar.gz
```

Verify integrity with:

```bash
sha256sum backup.tar.gz
sha256sum dataset_dollarstreet.tar.gz
```

Expected SHA-256 values:

- `backup.tar.gz`: `94ba8c7e4c5413c5db31e6b8fa88c648d6bc0fe1e82ab2642398dda719396cff`
- `dataset_dollarstreet.tar.gz`: `cff780f4418bd40ae60aa7ed3a7fa3619d25de30e2178c94f8e63b9b1c92a97f`

## Repository Status

This checkout contains the top-level analysis and generation scripts plus the recovered `evals/` helper scripts and example edit config, but it still does not include the full `results/`, `corrected/`, or `marvl/` trees that several scripts expect.

To fully reproduce the paper, you need both:

- this repository checkout
- the local backup assets generated during the original project run

The main missing reproducibility pieces are still the processed artifacts under `corrected/`, plus the raw local input trees under `results/` and `marvl/`. Those artifacts should be published in the repo, a release, or a companion archive if you want third parties to reproduce the paper end-to-end without private local state.

## Environment

Use Python 3.10 or 3.11. A CUDA GPU is effectively required for the LLaVA and image-editing scripts.

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

Copy the example environment file and edit it for your machine:

```bash
cp env.example.sh env.sh
source env.sh
mkdir -p "$CROSSROADS_DOLLARSTREET_CACHE" "$CROSSROADS_MODEL_CACHE" "$CROSSROADS_INSTRUCTBLIP_CACHE" "$CROSSROADS_EDITS_CACHE"
```

Path variables used by the repo:

- `CROSSROADS_DATA_ROOT`: extracted project-data root; defaults to the repo root
- `CROSSROADS_SECRETS_FILE`: path to `secrets.json`
- `CROSSROADS_DOLLARSTREET_CACHE`: Hugging Face cache for the Dollar Street dataset
- `CROSSROADS_MODEL_CACHE`: cache root for LLaVA and related model weights
- `CROSSROADS_INSTRUCTBLIP_CACHE`: cache root for InstructBLIP weights
- `CROSSROADS_EDITS_CACHE`: cache root for editing and similarity models

## Local Files You Need

Create a `secrets.json` file in the repo root:

```json
{
  "AZURE_OPENAI_ENDPOINT": "...",
  "AZURE_OPENAI_API_KEY": "...",
  "GPT4V_OPENAI_ENDPOINT": "...",
  "GPT4V_OPENAI_API_KEY": "..."
}
```

The scripts assume the following local inputs exist under `CROSSROADS_DATA_ROOT`:

- `results/dalle_images.csv` for DALLE Street classification and object extraction
- `marvl/marvl_images.csv` plus the underlying MARVL image files for the MARVL scripts
- `results/human-study/study1/*.csv` and `results/human-study/study2/*.csv` for `human_evals.py`
- `corrected/*.csv` and `corrected/*.json` for the plotting and summary scripts
- `results/dalle_objects/` intermediates for the artifact-extraction pipeline

If you extract the release tarballs somewhere other than the repo root, set `CROSSROADS_DATA_ROOT` to that extracted directory before running the scripts.

## Reproduce From Scratch

If you want the original full pipeline, use this order.

Recommended setup flow for a clean machine:

1. Clone the repo.
2. Download and reconstruct the release tarballs.
3. Extract the tarballs.
4. Set `CROSSROADS_DATA_ROOT` to the extracted data tree if it is not the repo root.
5. Create `secrets.json` with your own credentials.
6. Source `env.sh`.
7. Run the scripts below.

### 1. Generate DALLE Street images

This step creates the synthetic image set used later by the DALLE Street evaluation scripts.

```bash
./.venv/bin/python gen_dalle_street.py --category car
```

The script currently overrides `--category` internally and loops over all categories, writing images under `results/dalle_natural/<category>/<country>/`.

### 2. Build or restore `results/dalle_images.csv`

Several scripts require a manifest at `results/dalle_images.csv`. If you have the old local backup, restore that file first. If not, you will need to recreate a manifest that maps each generated image to:

- `image_path`
- `country`
- `concept`
- `type`

### 3. Run Dollar Street geographic classification

GPT-4V / Azure OpenAI:

```bash
for split in car cups_mugs_glasses family_snapshots front_door home kitchen plate_of_food social_drink wall_decoration wardrobe; do
  ./.venv/bin/python classify_dollar_street_gpt.py --split "$split"
done
```

LLaVA:

```bash
for split in car cups_mugs_glasses family_snapshots front_door home kitchen plate_of_food social_drink wall_decoration wardrobe; do
  ./.venv/bin/python classify_dollar_street_llava.py --split "$split"
done
```

### 4. Run DALLE Street geographic classification

GPT-4V / Azure OpenAI:

```bash
for kind in natural vivid; do
  for concept in car cups_mugs_glasses family_snapshots front_door home kitchen plate_of_food social_drink wall_decoration wardrobe; do
    ./.venv/bin/python classify_dalle_street_gpt.py --type "$kind" --concept "$concept"
  done
done
```

LLaVA:

```bash
for kind in natural vivid; do
  for concept in car cups_mugs_glasses family_snapshots front_door home kitchen plate_of_food social_drink wall_decoration wardrobe; do
    ./.venv/bin/python classify_dalle_street_llava.py --type "$kind" --concept "$concept"
  done
done
```

### 5. Run MARVL geographic classification

These scripts require `marvl/marvl_images.csv` and the MARVL image tree.

GPT-4V / Azure OpenAI:

```bash
for lang in id sw ta tr zh; do
  ./.venv/bin/python classify_marvl_gpt.py --country "$lang"
done
```

LLaVA:

```bash
for lang in id sw ta tr zh; do
  ./.venv/bin/python classify_marvl_llava.py --country "$lang"
done
```

### 6. Normalize the raw classification outputs

Dollar Street and DALLE Street:

```bash
./.venv/bin/python classify_evals.py
```

MARVL:

```bash
./.venv/bin/python marvl_evals.py
```

### 7. Aggregate headline accuracy numbers

```bash
./.venv/bin/python overall_acc.py
```

### 8. Render paper plots

Main Dollar Street / DALLE Street plots:

```bash
./.venv/bin/python plot.py
```

MARVL plots:

```bash
./.venv/bin/python plot_marvl.py
```

Additional dataset and artifact statistics:

```bash
./.venv/bin/python data_stats.py
./.venv/bin/python human_evals.py
./.venv/bin/python rgb_calculate.py
./.venv/bin/python people_counter.py
```

## Artifact Extraction Pipeline

These scripts support the artifact-analysis part of the paper and assume that `results/dalle_images.csv` and the corresponding image/result trees already exist.

Run them in this order:

1. `dalle_obj_detect.py`
2. `dalle_obj_process.py`
3. `dalle_obj_det_process.py`
4. `dalle_obj_counts.py`
5. `country_obj_dict_remove_adj.py`
6. `country_obj_cooccur.py`
7. `coocurrence_statistics.py`

Expected intermediate files include:

- `results/dalle_objects/*.csv`
- `results/dalle_objects/country_dict.json`
- `corrected/objects_proc.csv`
- `corrected/objects_proc_filtered.csv`
- `corrected/country_obj_*.json`
- `corrected/country_obj_*.csv`

## Suggested Script Order

This is the numbered ordering of the active top-level scripts in the current repo.

1. `gen_dalle_street.py`
2. `classify_dollar_street_gpt.py`
3. `classify_dollar_street_llava.py`
4. `classify_dalle_street_gpt.py`
5. `classify_dalle_street_llava.py`
6. `classify_marvl_gpt.py`
7. `classify_marvl_llava.py`
8. `classify_evals.py`
9. `marvl_evals.py`
10. `overall_acc.py`
11. `plot.py`
12. `plot_marvl.py`
13. `dalle_obj_detect.py`
14. `dalle_obj_process.py`
15. `dalle_obj_det_process.py`
16. `dalle_obj_counts.py`
17. `country_obj_dict_remove_adj.py`
18. `country_obj_cooccur.py`
19. `coocurrence_statistics.py`
20. `data_stats.py`
21. `human_evals.py`
22. `people_counter.py`
23. `rgb_calculate.py`
24. `dalle_street_hf_upload.py`
25. `edit.py`

## Practical Reproduction Notes

- `plot.py`, `plot_marvl.py`, `overall_acc.py`, and `data_stats.py` depend on processed `corrected/` outputs, not just raw model generations.
- The repo does not currently ship those processed artifacts, so figure reproduction is incomplete without the old backup.
- `edit.py` is optional and belongs to the image-editing experiments rather than the main geographic-classification pipeline.
- `dalle_street_hf_upload.py` is only needed if you want to package and publish the DALLE Street dataset after generation.
