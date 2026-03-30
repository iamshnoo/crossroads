import os
import sys
import argparse
import yaml
from tqdm import tqdm

from pnp_diffusers.pnp import PNP
from pnp_diffusers.pnp_utils import seed_everything

def run_pnp_for_configs(configs_dir):
    """
    Runs the PNP image editing process for all config files in the specified directory.

    Args:
        configs_dir (str): Directory containing config YAML files.
    """
    # Collect all config YAML files
    config_files = []
    for root, dirs, files in os.walk(configs_dir):
        for file in files:
            if file.endswith('.yaml'):
                config_files.append(os.path.join(root, file))

    if not config_files:
        print(f"No config files found in {configs_dir}")
        return

    for config_path in tqdm(config_files):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Ensure output directory exists
            os.makedirs(config["output_path"], exist_ok=True)

            # Save the config to the output directory
            with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
                yaml.dump(config, f)

            seed_everything(config["seed"])
            print(f"Running PNP for config: {config_path}")
            pnp = PNP(config)
            pnp.run_pnp()
        except Exception as e:
            print(f"Error processing config {config_path}: {e}")

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

    for CONCEPT in tqdm(concepts, total=len(concepts)):
        configs_dir = f"results/cap_edit/pnp-configs/{CONCEPT}/{args.src_country}_to_{args.tgt_country}"
        if not os.path.exists(configs_dir):
            print(f"Configs directory does not exist: {configs_dir}")
            continue
        run_pnp_for_configs(configs_dir)
