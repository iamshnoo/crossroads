import os
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchmetrics.functional.multimodal import clip_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import data_root

def compute_clip_scores(image1, image2, country1, country2):
    """Compute and return the CLIP score deltas for two images."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image1_resized = image1.resize((224, 224))
    image2_resized = image2.resize((224, 224))

    image1_tensor = (
        torch.from_numpy(np.array(image1_resized))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )
    image2_tensor = (
        torch.from_numpy(np.array(image2_resized))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    i1_country1 = clip_score(
        image1_tensor, country1, model_name_or_path="openai/clip-vit-base-patch16"
    ).item()
    i1_country2 = clip_score(
        image1_tensor, country2, model_name_or_path="openai/clip-vit-base-patch16"
    ).item()
    i2_country1 = clip_score(
        image2_tensor, country1, model_name_or_path="openai/clip-vit-base-patch16"
    ).item()
    i2_country2 = clip_score(
        image2_tensor, country2, model_name_or_path="openai/clip-vit-base-patch16"
    ).item()

    delta1 = i2_country1 - i1_country1  # Change in score for country1 (source)
    delta2 = i2_country2 - i1_country2  # Change in score for country2 (target)
    return delta1, delta2

def evaluate_pnp_results(captions_csv, base_image_path, pnp_output_base_path, output_csv, src_country, tgt_country):
    """
    Evaluates the PNP results by computing CLIP score deltas and stores the results in a CSV.

    Args:
        captions_csv (str): Path to the CSV file containing the image paths.
        base_image_path (str): Base path where the original images are stored.
        pnp_output_base_path (str): Base path where the PNP edited images are stored.
        output_csv (str): Path to save the evaluation results CSV.
        src_country (str): Source country name.
        tgt_country (str): Target country name.
    """
    # Read the CSV containing the images
    df = pd.read_csv(captions_csv)
    results = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_path_relative = row["image_path"]
        image_id = image_path_relative.split("/")[-1].split(".")[0]
        category = row["category"]
        country_row = row["country"]

        # Ensure the country matches the specified source country
        if country_row != src_country:
            continue

        # Paths
        original_image_path = os.path.join(base_image_path, image_path_relative)
        pnp_output_dir = os.path.join(pnp_output_base_path, category, f"{src_country}_to_{tgt_country}", image_id)

        if not os.path.exists(pnp_output_dir):
            print(f"PNP output directory does not exist: {pnp_output_dir}")
            continue
        # Find the edited image file in the output directory
        edited_image_files = [f for f in os.listdir(pnp_output_dir) if f.startswith("output") and f.endswith(".png")]
        if not edited_image_files:
            print(f"No edited image found in {pnp_output_dir}")
            continue
        edited_image_path = os.path.join(pnp_output_dir, edited_image_files[0])

        # Check if the edited image exists
        if not os.path.exists(edited_image_path):
            print(f"Edited image does not exist: {edited_image_path}")
            continue

        try:
            # Load images
            original_image = Image.open(original_image_path).convert("RGB")
            edited_image = Image.open(edited_image_path).convert("RGB")

            # Compute CLIP score deltas
            delta_source, delta_target = compute_clip_scores(original_image, edited_image, src_country, tgt_country)

            # Append results
            results.append({
                "image_id": image_id,
                "category": category,
                "source_country": src_country,
                "target_country": tgt_country,
                "original_image_path": image_path_relative,
                "edited_image_path": edited_image_path,
                "clip_score_delta_source": delta_source,
                "clip_score_delta_target": delta_target,
            })

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            continue

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_country", type=str, required=True, help="Source country name.")
    parser.add_argument("--tgt_country", type=str, required=True, help="Target country name.")
    args = parser.parse_args()

    if args.src_country == "UnitedStates":
        args.src_country = "United States"
    if args.tgt_country == "UnitedStates":
        args.tgt_country = "United States"

    concepts = [
        "car",
        "cups_mugs_glasses",
        "family_snapshots",
        "front_door",
        "home",
        "kitchen",
        "plate_of_food",
        "social_drink",
        "wall_decoration",
        "wardrobe",
    ]

    INPUT_DIR = "results/cap_edit/edited_captions"
    BASE_IMAGE_PATH = str(data_root())
    PNP_OUTPUT_BASE_PATH = "results/cap_edit/pnp-results"
    OUTPUT_DIR = "results/cap_edit/metrics"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Loop over each concept
    for CONCEPT in tqdm(concepts, total=len(concepts)):
        captions_csv = os.path.join(INPUT_DIR, f"{args.src_country}_to_{args.tgt_country}_edited_captions.csv")
        if not os.path.exists(captions_csv):
            print(f"Edited captions CSV does not exist: {captions_csv}")
            continue

        output_csv = os.path.join(OUTPUT_DIR, f"{CONCEPT}_{args.src_country}_to_{args.tgt_country}_evaluation.csv")

        evaluate_pnp_results(
            captions_csv=captions_csv,
            base_image_path=BASE_IMAGE_PATH,
            pnp_output_base_path=PNP_OUTPUT_BASE_PATH,
            output_csv=output_csv,
            src_country=args.src_country,
            tgt_country=args.tgt_country
        )
