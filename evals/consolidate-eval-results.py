# consolidate_metrics.py

import os
import pandas as pd
from tqdm import tqdm
import itertools

def consolidate_metrics(pnp_metrics_dir, cultureadapt_metrics_dir, output_csv):
    """
    Consolidates all metric CSV files from PNP and CultureAdapt into a single CSV file.
    """
    consolidated_data = []

    # Process PNP metrics
    print("Processing PNP metrics...")
    for root, dirs, files in os.walk(pnp_metrics_dir):
        for file in tqdm(files):
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)

                    # Extract information from the file path
                    # Assuming file name format: {concept}_{src_country}_to_{tgt_country}_evaluation.csv
                    file_name = os.path.basename(file_path)
                    name_parts = file_name.replace('.csv', '').split('_')
                    concept = name_parts[0]
                    src_country = name_parts[1]
                    tgt_country = name_parts[3]

                    # Add extracted information as new columns
                    df['method'] = 'PNP'
                    df['concept'] = concept
                    df['source_country'] = src_country
                    df['target_country'] = tgt_country

                    # Extract image_id from 'original_image_path' if 'image_id' is missing or empty
                    if 'image_id' not in df.columns or df['image_id'].isnull().all():
                        df['image_id'] = df['original_image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

                    # Extract 'original_image_source' from 'original_image_path'
                    df['original_image_source'] = df['original_image_path'].apply(lambda x: x.split('/')[1] if len(x.split('/')) > 1 else 'unknown')

                    consolidated_data.append(df)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Process CultureAdapt metrics
    print("Processing CultureAdapt metrics...")
    for root, dirs, files in os.walk(cultureadapt_metrics_dir):
        for file in tqdm(files):
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)

                    # Extract information from the file path
                    # Assuming file name format: {src_country}_to_{tgt_country}_results.csv
                    file_name = os.path.basename(file_path)
                    name_parts = file_name.replace('.csv', '').split('_')
                    src_country = name_parts[0]
                    tgt_country = name_parts[2]

                    # Add extracted information as new columns
                    df['method'] = 'CultureAdapt'
                    df['concept'] = df['category'] if 'category' in df.columns else 'Unknown'
                    df['source_country'] = src_country
                    df['target_country'] = tgt_country

                    # Rename columns to match PNP metrics for consistency
                    df.rename(columns={
                        'delta1_country1': 'clip_score_delta_source',
                        'delta2_country2': 'clip_score_delta_target',
                        'image_path': 'original_image_path',
                        'inpainting_path': 'edited_image_path'
                    }, inplace=True)

                    # Extract 'image_id' from 'original_image_path'
                    df['image_id'] = df['original_image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

                    # Extract 'original_image_source' from 'original_image_path'
                    df['original_image_source'] = df['original_image_path'].apply(lambda x: x.split('/')[1] if len(x.split('/')) > 1 else 'unknown')

                    consolidated_data.append(df)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Combine all dataframes
    if consolidated_data:
        consolidated_df = pd.concat(consolidated_data, ignore_index=True)

        # Remove duplicates based on 'image_id', 'method', 'source_country', 'target_country', 'concept'
        consolidated_df.drop_duplicates(subset=['image_id', 'method', 'source_country', 'target_country', 'concept'], inplace=True)

        # Save the consolidated DataFrame to CSV
        consolidated_df.to_csv(output_csv, index=False)
        print(f"Consolidated metrics saved to {output_csv}")
    else:
        print("No data to consolidate.")

if __name__ == "__main__":
    pnp_metrics_dir = "results/cap_edit/metrics"
    cultureadapt_metrics_dir = "results/cultureadapt/metrics"
    output_csv = "results/consolidated_metrics.csv"

    df = pd.read_csv("artifacts.csv")
    # extract image_ids from image_path
    image_ids = df['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    # add image_ids to df
    df['image_id'] = image_ids
    # print(df.head())

    countries = ["Brazil", "India", "Nigeria", "Turkey", "United States"]
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
    methods = ["ca", "pnp"]

    results = {}
    for source_country, target_country, concept, method in tqdm(itertools.product(countries, countries, concepts, methods), total=len(countries)*len(countries)*len(concepts)*len(methods)):
        src_country = source_country
        target_country = target_country
        concept = concept
        method = method
        relevant_ids = df[(df['country'] == src_country) & (df["concept"] == concept)]['image_id'].unique()
        uid_exp = f"{method}_{src_country}_to_{target_country}_{concept}"
        if method == "ca":
            for path in os.listdir(cultureadapt_metrics_dir):
                if path == f"{src_country}_to_{target_country}_results.csv":
                    print(path)
                    ca_metrics = pd.read_csv(os.path.join(cultureadapt_metrics_dir, path))
                    ca_metrics = ca_metrics[ca_metrics['category'] == concept]
                    ca_metrics["image_id"] = ca_metrics["image_path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
                    for image_id in relevant_ids:
                        uid = uid_exp + "_" + image_id
                        og_paths = ca_metrics[ca_metrics["image_id"] == image_id]["image_path"].values
                        for og_path in og_paths:
                            if "dalle_natural" in og_path:
                                uuid = uid + "_dalle_natural"
                                img_type = "dalle_natural"
                                edited_path = ca_metrics[(ca_metrics["image_id"] == image_id) & (ca_metrics["image_path"] == og_path)]["inpainting_path"].values[0]
                                delta1 = ca_metrics[(ca_metrics["image_id"] == image_id) & (ca_metrics["image_path"] == og_path)]["delta1_country1"].values[0]
                                delta2 = ca_metrics[(ca_metrics["image_id"] == image_id) & (ca_metrics["image_path"] == og_path)]["delta2_country2"].values[0]
                            else:
                                uuid = uid + "_dalle_vivid"
                                img_type = "dalle_vivid"
                                edited_path = ca_metrics[(ca_metrics["image_id"] == image_id) & (ca_metrics["image_path"] == og_path)]["inpainting_path"].values[0]
                                delta1 = ca_metrics[(ca_metrics["image_id"] == image_id) & (ca_metrics["image_path"] == og_path)]["delta1_country1"].values[0]
                                delta2 = ca_metrics[(ca_metrics["image_id"] == image_id) & (ca_metrics["image_path"] == og_path)]["delta2_country2"].values[0]
                            results[uuid] = {
                                "image_id": image_id,
                                "method": "cultureadapt",
                                "source_country": src_country,
                                "target_country": target_country,
                                "concept": concept,
                                "img_type": img_type,
                                "original_image_path": og_path,
                                "edited_image_path": edited_path,
                                "delta1": delta1,
                                "delta2": delta2,
                            }
                            # print(results[uuid])
                        # break
        elif method == "pnp":
            for path in os.listdir(pnp_metrics_dir):
                if path == f"{concept}_{src_country}_to_{target_country}_evaluation.csv":
                    print(path)
                    pnp_metrics = pd.read_csv(os.path.join(pnp_metrics_dir, path))
                    pnp_metrics = pnp_metrics[pnp_metrics['category'] == concept]
                    for image_id in relevant_ids:
                        uid = uid_exp + "_" + image_id
                        og_paths = pnp_metrics[pnp_metrics["image_id"] == image_id]["original_image_path"].values
                        for og_path in og_paths:
                            if "dalle_natural" in og_path:
                                uuid = uid + "_dalle_natural"
                                img_type = "dalle_natural"
                                edited_path = pnp_metrics[(pnp_metrics["image_id"] == image_id) & (pnp_metrics["original_image_path"] == og_path)]["edited_image_path"].values[0]
                                delta1 = pnp_metrics[(pnp_metrics["image_id"] == image_id) & (pnp_metrics["original_image_path"] == og_path)]["clip_score_delta_source"].values[0]
                                delta2 = pnp_metrics[(pnp_metrics["image_id"] == image_id) & (pnp_metrics["original_image_path"] == og_path)]["clip_score_delta_target"].values[0]
                            else:
                                uuid = uid + "_dalle_vivid"
                                img_type = "dalle_vivid"
                                edited_path = pnp_metrics[(pnp_metrics["image_id"] == image_id) & (pnp_metrics["original_image_path"] == og_path)]["edited_image_path"].values[0]
                                delta1 = pnp_metrics[(pnp_metrics["image_id"] == image_id) & (pnp_metrics["original_image_path"] == og_path)]["clip_score_delta_source"].values[0]
                                delta2 = pnp_metrics[(pnp_metrics["image_id"] == image_id) & (pnp_metrics["original_image_path"] == og_path)]["clip_score_delta_target"].values[0]
                            results[uuid] = {
                                "image_id": image_id,
                                "method": "cap-edit",
                                "source_country": src_country,
                                "target_country": target_country,
                                "concept": concept,
                                "img_type": img_type,
                                "original_image_path": og_path,
                                "edited_image_path": edited_path,
                                "delta1": delta1,
                                "delta2": delta2,
                            }
                            # print(results[uuid])
                        # break
                        # results[uuid] = {
                        #     "image_id": image_id,
                        #     "method": "cap-edit",
                        #     "source_country": src_country,
                        #     "target_country": target_country,
                        #     "concept": concept,
                        #     "original_image_path": pnp_metrics[pnp_metrics["image_id"] == image_id]["original_image_path"].values[0],
                        #     "edited_image_path": pnp_metrics[pnp_metrics["image_id"] == image_id]["edited_image_path"].values[0],
                        #     "delta1": pnp_metrics[pnp_metrics["image_id"] == image_id]["clip_score_delta_source"].values[0],
                        #     "delta2": pnp_metrics[pnp_metrics["image_id"] == image_id]["clip_score_delta_target"].values[0],
                        # }

    # print(results)




    # change results to a df
    results = pd.DataFrame.from_dict(results, orient='index')
    results.to_csv("results/consolidated_metrics.csv")
    # results.to_csv("results/cultureadapt_metrics.csv")
    # results.to_csv("results/pnp_metrics.csv")



    # consolidate_metrics(pnp_metrics_dir, cultureadapt_metrics_dir, output_csv)
