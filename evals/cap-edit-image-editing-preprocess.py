import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from argparse import Namespace
import argparse

from pnp_diffusers.preprocess import run

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import data_root

def process_images(artifacts_csv, base_path, save_dir, country):
    artifacts = pd.read_csv(artifacts_csv)
    # Filter artifacts for the specified country
    artifacts = artifacts[artifacts["image_path"].str.contains(country)]

    for index, row in tqdm(artifacts.iterrows(), total=len(artifacts)):
        image_path_relative = row["image_path"]
        image_id = image_path_relative.split("/")[-1].split(".")[0] #os.path.splitext(os.path.basename(image_path_relative))[0]
        category = row["concept"]
        country_row = row["country"]

        # Ensure the country matches the specified country
        if country_row != country:
            continue

        # Full image path
        image_path = os.path.join(base_path, image_path_relative)

        try:
            # Create options for the run function
            opt = Namespace(
                data_path=image_path,
                save_dir=os.path.join(save_dir, category, country, image_id),
                sd_version="2.1",
                seed=1,
                steps=999,
                save_steps=1000,
                inversion_prompt="",
                extract_reverse=False,
            )

            # Ensure the save directory exists
            os.makedirs(opt.save_dir, exist_ok=True)

            # Run the preprocess function
            run(opt)
        except Exception as e:
            print(f"Error processing {image_path_relative}: {e}")

if __name__ == "__main__":
    # List of countries to process
    countries = ["Brazil", "India", "Nigeria", "Turkey", "United States"]

    # accept country as a command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, required=True)
    args = parser.parse_args()
    if args.country == "UnitedStates":
        args.country = "United States"

    BASE_PATH = str(data_root())
    SAVE_DIR = "results/cap_edit/latents"
    COUNTRY = args.country
    print(f"Processing image latents for {COUNTRY}")
    process_images(
        artifacts_csv="artifacts.csv",
        base_path=BASE_PATH,
        save_dir=SAVE_DIR,
        country=COUNTRY
    )


    # Process images for each country
    # for country in tqdm(countries, total=len(countries)):
    #     process_images(
    #         artifacts_csv="artifacts.csv",
    #         base_path=BASE_PATH,
    #         save_dir=SAVE_DIR,
    #         country=country
    #     )
