# import os
# import json
# import pandas as pd
# from openai import OpenAI
# from tqdm import tqdm

# with open("secrets.json", "r") as f:
#     secrets = json.load(f)

# client = OpenAI(api_key = secrets["OPENAI_API_KEY"])

# # Function to call OpenAI API
# def answer_fn(prompt):
#     response = client.completions.create(
#         model="gpt-3.5-turbo-instruct",
#         prompt=prompt,
#         temperature=0.7,
#         max_tokens=256,
#         top_p=0.5,
#         frequency_penalty=0,
#         presence_penalty=0,
#         n=1,
#     )
#     return response.choices[0].text.strip()

# # Function to edit captions
# def edit_captions(captions_csv, output_csv, country_name):
#     df = pd.read_csv(captions_csv)
#     edited_captions = []
#     llm_prompt = f'Edit the input text, such that it is culturally relevant to {country_name}. Keep the output text of a similar length as the input text. If it is already culturally relevant to {country_name}, no need to make any edits. The output text must be in English only.\nInput: '

#     for index, row in tqdm(df.iterrows(), total=len(df)):
#         caption = row['caption']
#         prompt = llm_prompt + caption + "\nOutput: "
#         try:
#             gen_text = answer_fn(prompt)
#             # Clean up the generated text
#             for char in ['"', ';', '.']:
#                 gen_text = gen_text.replace(char, "")
#             edited_captions.append(gen_text)
#         except Exception as e:
#             print(f"Error processing caption: {e}")
#             edited_captions.append(caption)  # Keep the original caption if there's an error

#     # Add edited captions to the DataFrame
#     df['edited_caption'] = edited_captions
#     df.to_csv(output_csv, index=False)

# if __name__ == "__main__":
#     # List of countries to process
#     countries = ["Brazil", "India", "Nigeria", "Turkey", "United States"]
#     INPUT_DIR = "results/cap_edit/instruct_blip"
#     OUTPUT_DIR = "results/cap_edit/edited_captions"
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # Process captions for each country
#     for country in tqdm(countries, total=len(countries)):
#         captions_csv = os.path.join(INPUT_DIR, f"{country}_captions.csv")
#         output_csv = os.path.join(OUTPUT_DIR, f"{country}_edited_captions.csv")
#         edit_captions(captions_csv, output_csv, country)


import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

with open("secrets.json", "r") as f:
    secrets = json.load(f)

client = OpenAI(api_key = secrets["OPENAI_API_KEY"])

# Function to call OpenAI API
def answer_fn(prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
    )
    return response.choices[0].text.strip()

# Function to edit captions
def edit_captions(captions_csv, output_csv, src_country, tgt_country):
    df = pd.read_csv(captions_csv)
    edited_captions = []
    if src_country == "United States":
        src_country = "the United States"
    if tgt_country == "United States":
        tgt_country = "the United States"
    llm_prompt = f'Edit the input text, which is relevant to {src_country}, to make it culturally relevant to {tgt_country}. Keep the output text of a similar length as the input text. The output text must be in English only.\nInput: '

    for index, row in tqdm(df.iterrows(), total=len(df)):
        caption = row['caption']
        prompt = llm_prompt + caption + "\nOutput: "
        try:
            gen_text = answer_fn(prompt)
            # Clean up the generated text
            for char in ['"', ';', '.']:
                gen_text = gen_text.replace(char, "")
            edited_captions.append(gen_text)
        except Exception as e:
            print(f"Error processing caption: {e}")
            edited_captions.append(caption)  # Keep the original caption if there's an error

    # Add edited captions and country info to the DataFrame
    df['edited_caption'] = edited_captions
    df['source_country'] = src_country
    df['target_country'] = tgt_country
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # List of countries to process
    countries = ["Brazil", "India", "Nigeria", "Turkey", "United States"]
    INPUT_DIR = "results/cap_edit/instruct_blip"
    OUTPUT_DIR = "results/cap_edit/edited_captions"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create source-target country pairs excluding where source == target
    country_pairs = [(src, tgt) for src in countries for tgt in countries if src != tgt]

    # Process captions for each source-target country pair
    for src_country, tgt_country in tqdm(country_pairs, total=len(country_pairs)):
        captions_csv = os.path.join(INPUT_DIR, f"{src_country}_captions.csv")
        output_csv = os.path.join(OUTPUT_DIR, f"{src_country}_to_{tgt_country}_edited_captions.csv")
        edit_captions(captions_csv, output_csv, src_country, tgt_country)
