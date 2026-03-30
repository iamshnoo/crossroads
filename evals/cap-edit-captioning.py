import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import data_root, instructblip_cache_dir

# Model setup
model_name = "Salesforce/instructblip-flan-t5-xxl"
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CACHE_DIR = instructblip_cache_dir()

model = InstructBlipForConditionalGeneration.from_pretrained(
    model_name, cache_dir=str(MODEL_CACHE_DIR)
).to(device)
processor = InstructBlipProcessor.from_pretrained(
    model_name, cache_dir=str(MODEL_CACHE_DIR)
)


# Function to generate caption for a single image
def generate_caption(image):
    prompt = "A short image description:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=0.9,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[
            0
    ].strip()
    return generated_text


# Function to process artifacts and generate captions
def process_artifacts(artifacts_csv, base_path, output_csv, country):
    artifacts = pd.read_csv(artifacts_csv)
    # Filter artifacts for the specified country
    artifacts = artifacts[artifacts["image_path"].str.contains(country)]
    results = []

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
            # Load image
            image = Image.open(image_path).convert("RGB")
            # Generate caption
            caption = generate_caption(image)
            # Append result to list
            results.append(
                {
                    "image_path": image_path_relative,
                    "category": category,
                    "country": country,
                    "caption": caption,
                }
            )
        except Exception as e:
            print(f"Error processing {image_path_relative}: {e}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    # List of countries to process
    countries = ["Brazil", "India", "Nigeria", "Turkey", "United States"]
    BASE_PATH = str(data_root())
    OUTPUT_DIR = "results/cap_edit/instruct_blip"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process artifacts for each country
    for country in countries:
        output_csv = os.path.join(OUTPUT_DIR, f"{country}_captions.csv")
        process_artifacts(
            artifacts_csv="artifacts.csv",
            base_path=BASE_PATH,
            output_csv=output_csv,
            country=country,
        )
