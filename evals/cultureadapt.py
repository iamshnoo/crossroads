import os
import ast
import sys
from pathlib import Path

import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from diffusers import (
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from torchmetrics.functional.multimodal import clip_score
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import data_root, edits_cache_dir

sns.set(font_scale=1.4)
sns.set_theme(style="whitegrid")
sns.set_context(
    "paper", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20}
)

# # Model setup
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = edits_cache_dir()

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=str(CACHE_DIR),
    clean_up_tokenization_spaces=True,
)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    model_id, cache_dir=str(CACHE_DIR)
).to(device)

# Load the inpainting pipeline and VAE with proper configuration
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    cache_dir=str(CACHE_DIR),
).to(device)
pipe.enable_xformers_memory_efficient_attention()

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16,
    cache_dir=str(CACHE_DIR),
    clean_up_tokenization_spaces=True,
).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, cache_dir=str(CACHE_DIR)
)
pipe.vae = vae


# Functions for the steps
def process_objects(artifact_str):
    """Convert artifacts to lowercase and split items joined by underscore."""
    objects_list = ast.literal_eval(artifact_str)

    # Ensure each element is a string and handle cases where it's a list
    processed_objects = []
    for obj in objects_list:
        if isinstance(obj, str):  # If it's a string, split and process
            processed_objects.append(" ".join(obj.split("_")).lower())
        elif isinstance(obj, list):  # If it's a list, join the list into a string
            processed_objects.append(" ".join(obj).lower())

    # Join all processed objects into a single string with periods
    objects = ". ".join(processed_objects)
    return objects + "."


def object_detection(image, objects, image_path, country1, category):
    """Run object detection and save annotated image."""
    text = f"{country1.lower()} " + objects
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )[0]

    # Save object detection result
    image_id = image_path.split("/")[-1].split(".")[0]
    output_path = (
        f"results/cultureadapt/dino/bb/{category}/{country1}/{image_id}_annotated.jpg"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_results(
        image,
        results["scores"].cpu(),
        results["labels"],
        results["boxes"].cpu(),
        output_path,
    )
    return results["boxes"].cpu(), output_path


def plot_results(pil_img, scores, labels, boxes, output_path):
    """Visualize and save object detection results."""
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]] * 100

    for score, text, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        label = f"{text}: {score:.2f}"
        ax.text(
            xmin, ymin, label, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5)
        )
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()


def generate_masks_with_grounding(image_source, boxes, image_path, country1, category):
    """Generate masks based on object detection boxes."""
    image_array = np.array(image_source)
    mask = np.zeros_like(image_array)
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        mask[ymin:ymax, xmin:xmax, :] = 255
    masked_image = np.where(mask == 255, 255, image_array)
    masked_image = Image.fromarray(masked_image)

    # Save masked image
    image_id = image_path.split("/")[-1].split(".")[0]
    mask_path = f"results/cultureadapt/dino/masked_images/{category}/{country1}/{image_id}_masked.jpg"
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    masked_image.save(mask_path)
    return masked_image


def run_inpainting(
    image, masked_image, objects, country2, category, image_path, country1
):
    """Run inpainting on the masked image and save the output."""
    prompt = (
        f"{country2.lower()} "
        + objects
        + " intricate details. 4k. high resolution. high quality."
    )
    image_source_for_inpaint = image.resize((512, 512))
    masked_image_for_inpaint = masked_image.resize((512, 512))
    generator = torch.Generator(device).manual_seed(random.randrange(0, 100000))

    inpainting_result = pipe(
        prompt=prompt,
        image=image_source_for_inpaint,
        mask_image=masked_image_for_inpaint,
        generator=generator,
    ).images[0]

    inpainting_result_resized = inpainting_result.resize(image.size)

    # Save inpainting result
    image_id = image_path.split("/")[-1].split(".")[0]
    output_path = f"results/cultureadapt/edits/sd2/{category}/{image_id}_inpainting_result_{country1}_to_{country2}.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    inpainting_result_resized.save(output_path)
    return inpainting_result_resized, output_path


def compute_clip_scores(image1, image2, country1, country2):
    """Compute and return the CLIP score deltas for two images."""
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

    delta1 = i2_country1 - i1_country1
    delta2 = i2_country2 - i1_country2
    return delta1, delta2


def process_artifacts(
    artifacts_csv, base_path, output_csv, country_src="India", country_target="Greece"
):
    """Process all query paths in artifacts.csv and save results."""
    artifacts = pd.read_csv(artifacts_csv)
    artifacts = artifacts[artifacts["image_path"].str.contains(country_src)]
    # artifacts = artifacts[:10]
    # print(artifacts.head())
    results = []

    for index, row in tqdm(artifacts.iterrows(), total=len(artifacts)):
        query_path = row["image_path"]
        image_id = query_path.split("/")[-1].split(".")[0]
        category = row["concept"]
        country1 = row["country"]
        country2 = country_target

        assert (
            country1 == country_src
        ), f"Source Country mismatch: {country1} != {country_src}"
        # Load image
        image_path = os.path.join(base_path, query_path)
        image = Image.open(image_path).convert("RGB")

        # Process objects
        artifact_str = row["artifacts"]
        try:
            objects = process_objects(artifact_str)

            # Object detection and mask generation
            boxes, annotation_output_path = object_detection(
                image, objects, query_path, country1, category
            )
            masked_image = generate_masks_with_grounding(
                image, boxes, query_path, country1, category
            )

            # Inpainting and saving result
            inpainted_image, inpainted_output_path = run_inpainting(
                image, masked_image, objects, country2, category, query_path, country1
            )

            # Compute CLIP score deltas
            delta1, delta2 = compute_clip_scores(
                image, inpainted_image, country1, country2
            )

            # Append results for final CSV
            results.append(
                {
                    "image_path": query_path,
                    "category": category,
                    "country1": country1,
                    "country2": country2,
                    "objects": artifact_str,
                    "annotated_path": annotation_output_path,
                    "inpainting_path": inpainted_output_path,
                    "delta1_country1": delta1,
                    "delta2_country2": delta2,
                }
            )
        except Exception as e:
            print(f"Error processing {query_path}: {e}")

    # Save results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    countries = ["Brazil", "India", "Nigeria", "Turkey", "United States"]
    country_matrix = [
        (src, tgt) for src in countries for tgt in countries if src != tgt
    ]
    # country_matrix = [
    #     ("Brazil", "India"),
    #     ("Brazil", "Nigeria"),
    #     ("Brazil", "Turkey"),
    #     ("Brazil", "United States"),
    #     ("India", "Brazil"),
    #     ("India", "Nigeria"),
    #     ("India", "Turkey"),
    #     ("India", "United States"),
    #     ("Nigeria", "Brazil"),
    #     ("Nigeria", "India"),
    #     ("Nigeria", "Turkey"),
    #     ("Nigeria", "United States"),
    #     ("Turkey", "Brazil"),
    #     ("Turkey", "India"),
    #     ("Turkey", "Nigeria"),
    #     ("Turkey", "United States"),
    #     ("United States", "Brazil"),
    #     ("United States", "India"),
    #     ("United States", "Nigeria"),
    #     ("United States", "Turkey"),
    # ]

    BASE_PATH = str(data_root())
    # print(country_matrix)

    # Run the process on all entries in artifacts.csv for a particular SOURCE, TARGET pair
    for COUNTRY_SRC, COUNTRY_TARGET in country_matrix:
        print(f"Processing {COUNTRY_SRC} to {COUNTRY_TARGET}")
        process_artifacts(
            "artifacts.csv",
            BASE_PATH,
            f"results/cultureadapt/metrics/{COUNTRY_SRC}_to_{COUNTRY_TARGET}_results.csv",
            COUNTRY_SRC,
            COUNTRY_TARGET,
        )

    # # Run the process on all entries in artifacts.csv for a particular SOURCE, TARGET pair
    # COUNTRY_SRC = "India"
    # COUNTRY_TARGET = "Greece"
    # process_artifacts("artifacts.csv", BASE_PATH, f"results/cultureadapt/{COUNTRY_SRC}_to_{COUNTRY_TARGET}_results.csv", COUNTRY_SRC, COUNTRY_TARGET)
