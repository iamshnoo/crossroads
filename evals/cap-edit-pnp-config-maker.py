# import os
# import pandas as pd
# import yaml
# from tqdm import tqdm

# def create_configs(captions_csv, base_image_path, latents_base_path, config_output_dir, country):
#     """
#     Creates config YAML files for each image based on the edited captions.

#     Args:
#         captions_csv (str): Path to the CSV file containing edited captions.
#         base_image_path (str): Base path where images are stored.
#         latents_base_path (str): Base path where latents are stored.
#         config_output_dir (str): Directory to save the generated config files.
#         country (str): Country name to filter images.
#     """
#     # Read the CSV containing the edited captions
#     df = pd.read_csv(captions_csv)

#     for index, row in tqdm(df.iterrows(), total=len(df)):
#         image_path_relative = row["image_path"]
#         image_id = image_path_relative.split("/")[-1].split(".")[0]  #os.path.splitext(os.path.basename(image_path_relative))[0]
#         category = row["category"]
#         country_row = row["country"]
#         edited_caption = row["edited_caption"]

#         # Ensure the country matches the specified country
#         if country_row != country:
#             continue

#         # Paths
#         image_path = os.path.join(base_image_path, image_path_relative)
#         latents_path = os.path.join(latents_base_path, category, country, image_id+"_forward")
#         output_path = os.path.join("results/cap_edit/pnp-results", category, country, image_id)
#         config_path = os.path.join(config_output_dir, category, country, f"{image_id}.yaml")

#         # Ensure the config directory exists
#         os.makedirs(os.path.dirname(config_path), exist_ok=True)

#         # Create config dictionary
#         config = {
#             # General settings
#             'seed': 1,
#             'device': 'cuda',
#             'output_path': output_path,

#             # Data paths
#             'image_path': image_path,
#             'latents_path': latents_path,

#             # Diffusion parameters
#             'sd_version': '2.1',
#             'guidance_scale': 7.5,
#             'n_timesteps': 50,
#             'prompt': edited_caption,
#             'negative_prompt': 'ugly, blurry, low res, unrealistic',

#             # PNP injection thresholds
#             'pnp_attn_t': 0.5,
#             'pnp_f_t': 0.8,
#         }

#         # Write config to YAML file
#         with open(config_path, 'w') as f:
#             yaml.dump(config, f)

# if __name__ == "__main__":
#     # List of countries to process
#     countries = ["Brazil", "India", "Nigeria", "Turkey", "United States"] #["India", "Greece"]
#     INPUT_DIR = "results/cap_edit/edited_captions"
#     BASE_IMAGE_PATH = "/scratch/amukher6/dollar_street/"
#     LATENTS_BASE_PATH = "/scratch/amukher6/dollar_street/evals/results/cap_edit/latents/"
#     CONFIG_OUTPUT_DIR = "results/cap_edit/pnp-configs"

#     for country in countries:
#         captions_csv = os.path.join(INPUT_DIR, f"{country}_edited_captions.csv")
#         create_configs(
#             captions_csv=captions_csv,
#             base_image_path=BASE_IMAGE_PATH,
#             latents_base_path=LATENTS_BASE_PATH,
#             config_output_dir=CONFIG_OUTPUT_DIR,
#             country=country
#         )

# Updated config-maker.py

import os
import pandas as pd
import yaml
from tqdm import tqdm

def create_configs(captions_csv_pair, base_image_path, latents_base_path, config_output_dir, src_country, tgt_country):
    """
    Creates config YAML files for each image, transforming images from src_country to appear culturally relevant to tgt_country.

    Args:
        captions_csv_pair (str): Path to the CSV file containing edited captions for the source-target country pair.
        base_image_path (str): Base path where images are stored.
        latents_base_path (str): Base path where latents are stored.
        config_output_dir (str): Directory to save the generated config files.
        src_country (str): Source country name.
        tgt_country (str): Target country name.
    """
    # Read the CSV containing the edited captions for the source-target country pair
    df_pair = pd.read_csv(captions_csv_pair)

    for index, row in tqdm(df_pair.iterrows(), total=len(df_pair)):
        image_path_relative = row["image_path"]
        image_id = image_path_relative.split("/")[-1].split(".")[0]
        category = row["category"]
        country_row = row["country"]  # This should be the source country
        edited_caption = row["edited_caption"]

        # Ensure the country matches the specified source country
        if country_row != src_country:
            continue

        # Paths
        image_path = os.path.join(base_image_path, image_path_relative)
        latents_path = os.path.join(latents_base_path, category, src_country, image_id + "_forward")
        output_path = os.path.join("results/cap_edit/pnp-results", category, f"{src_country}_to_{tgt_country}", image_id)
        config_path = os.path.join(config_output_dir, category, f"{src_country}_to_{tgt_country}", f"{image_id}.yaml")

        # Ensure the config directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        # Create config dictionary
        config = {
            # General settings
            'seed': 1,
            'device': 'cuda',
            'output_path': output_path,

            # Data paths
            'image_path': image_path,
            'latents_path': latents_path,

            # Diffusion parameters
            'sd_version': '2.1',
            'guidance_scale': 7.5,
            'n_timesteps': 50,
            'prompt': edited_caption,
            'negative_prompt': 'ugly, blurry, low res, unrealistic',

            # PNP injection thresholds
            'pnp_attn_t': 0.5,
            'pnp_f_t': 0.8,

            # Additional info
            'source_country': src_country,
            'target_country': tgt_country,
            'category': category,
            'image_id': image_id,
        }

        # Write config to YAML file
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

if __name__ == "__main__":
    # List of countries to process
    countries = ["Brazil", "India", "Nigeria", "Turkey", "United States"]

    # Create source-target country pairs excluding where source == target
    country_pairs = [(src, tgt) for src in countries for tgt in countries if src != tgt]

    INPUT_DIR = "results/cap_edit/edited_captions"  # Directory containing the edited captions CSV files
    BASE_IMAGE_PATH = "/scratch/amukher6/dollar_street/"
    LATENTS_BASE_PATH = "/scratch/amukher6/dollar_street/evals/results/cap_edit/latents/"
    CONFIG_OUTPUT_DIR = "results/cap_edit/pnp-configs"

    for src_country, tgt_country in country_pairs:
        captions_csv_pair = os.path.join(INPUT_DIR, f"{src_country}_to_{tgt_country}_edited_captions.csv")
        if not os.path.exists(captions_csv_pair):
            print(f"Edited captions file does not exist: {captions_csv_pair}")
            continue
        create_configs(
            captions_csv_pair=captions_csv_pair,
            base_image_path=BASE_IMAGE_PATH,
            latents_base_path=LATENTS_BASE_PATH,
            config_output_dir=CONFIG_OUTPUT_DIR,
            src_country=src_country,
            tgt_country=tgt_country
        )
