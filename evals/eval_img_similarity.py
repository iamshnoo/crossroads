# import pandas as pd
# import numpy as np
# import os

# import PIL
# from PIL import Image

# df = pd.read_csv("results/pnp_metrics.csv")

# BASE_PATH = "/scratch/amukher6/dollar_street/"
# SOURCE_PATH = os.path.join("/scratch/amukher6/dollar_street/", df.iloc[0]["original_image_path"])
# TARGET_PATH = os.path.join("/scratch/amukher6/dollar_street/evals/", df.iloc[0]["edited_image_path"])

# def load_image(path):
#     image = PIL.Image.open(path).convert("RGB")
#     return image

# source_image = load_image(SOURCE_PATH)
# target_image = load_image(TARGET_PATH)

import os
import torch
from transformers import ViTImageProcessor, ViTModel
import PIL
from PIL import Image
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# Function to load and preprocess an image
def load_image(path):
    try:
        image = PIL.Image.open(path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image from {path}: {e}")
        return None

# Function to calculate the cosine similarity between two images
def calculate_image_similarity(source_image_path, target_image_path, model, processor, device):
    source_image = load_image(source_image_path)
    target_image = load_image(target_image_path)

    # Check if images were successfully loaded
    if source_image is None or target_image is None:
        return None

    # Preprocess images
    source_input = processor(source_image, return_tensors="pt").to(device)
    target_input = processor(target_image, return_tensors="pt").to(device)

    # Extract features
    with torch.no_grad():
        source_features = model(**source_input).last_hidden_state.mean(dim=1)
        target_features = model(**target_input).last_hidden_state.mean(dim=1)

    # Normalize features
    source_features = F.normalize(source_features, p=2, dim=1)
    target_features = F.normalize(target_features, p=2, dim=1)

    # Calculate cosine similarity
    cosine_similarity = F.cosine_similarity(source_features, target_features).item()

    return cosine_similarity

# Function to find similarity between images in a dataframe
def find_image_similarities(df, base_path, model, processor, device):
    cosine_similarities = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        source_image_path = os.path.join(base_path, row["original_image_path"])
        target_image_path = os.path.join(base_path, "evals", row["edited_image_path"])
        similarity = calculate_image_similarity(source_image_path, target_image_path, model, processor, device)
        if similarity is not None:
            cosine_similarities.append((source_image_path, target_image_path, similarity))

    return cosine_similarities

# Main function
def main(method_name="cap_edit"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model and processor
    processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8', cache_dir="/projects/antonis/anjishnu/edits")
    model = ViTModel.from_pretrained('facebook/dino-vitb8', cache_dir="/projects/antonis/anjishnu/edits").eval().to(device)

    # Load dataframe
    method = method_name
    df = pd.read_csv("results/pnp_metrics.csv") if method == "cap_edit" else pd.read_csv("results/cultureadapt_metrics.csv")
    sim_resuls_df = df.copy()

    base_path = "/scratch/amukher6/dollar_street/"
    similarities = find_image_similarities(df, base_path, model, processor, device)
    # for src, tgt, sim in similarities:
    #     print(f"Source: {src}, Target: {tgt}, Similarity: {sim}")

    # Save results to a CSV file
    sim_resuls_df["similarity"] = [sim for _, _, sim in similarities]
    print(sim_resuls_df)
    sim_resuls_df.to_csv("results/cap_edit/similarity/pnp_metrics_all.csv", index=False) if method == "cap_edit" else sim_resuls_df.to_csv("results/cultureadapt/similarity/cultureadapt_metrics_all.csv", index=False)

if __name__ == "__main__":
    # main("cap_edit")
    main("cultureadapt")
