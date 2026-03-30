import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("../results/dalle_objects/objects_parsed.csv")
print(df.iloc[0])

# Initialize an empty dictionary to store artifacts
artifacts_dict = {}

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    image = row["id"]
    true_country = row["true_country"]
    concept = row["concept"]
    image_type = row["type"]
    relevant_items = row["Relevant_Items"]
    non_relevant_items = row["Non_Relevant_Items"]

    # Initialize a new dictionary for the image if it doesn't exist
    if image not in artifacts_dict:
        artifacts_dict[image] = {}

    artifacts_dict[image]["country"] = true_country
    artifacts_dict[image]["concept"] = concept
    artifacts_dict[image]["type"] = image_type

    artifacts_dict[image]["artifacts"] = []
    # Process Relevant_Items
    for item in eval(
        relevant_items
    ):  # Assuming 'Relevant_Items' is a string representation of a list
        if "object" in item:
            artifacts_dict[image]["artifacts"].append(item["object"])
        elif "object_type" in item:
            artifacts_dict[image]["artifacts"].append(item["object_type"])
        elif "type" in item:
            artifacts_dict[image]["artifacts"].append(item["type"])

    # Process Non_Relevant_Items
    for item in eval(
        non_relevant_items
    ):  # Assuming 'Non_Relevant_Items' is a string representation of a list
        if "object" in item:
            artifacts_dict[image]["artifacts"].append(item["object"])
        elif "object_type" in item:
            artifacts_dict[image]["artifacts"].append(item["object_type"])
        elif "type" in item:
            artifacts_dict[image]["artifacts"].append(item["type"])

# Print the flattened list for the given image
print(artifacts_dict["results/dalle_natural/car/Austria/Austria_1.jpg"])

# save to csv
df = pd.DataFrame(artifacts_dict).T
# name the index column
df.index.name = "image_path"
df.to_csv("artifacts.csv")

# import ast
# BASE_PATH = "/scratch/amukher6/dollar_street/"
# QUERY_PATH = "results/dalle/plate_of_food/Vietnam/Vietnam_3.jpg"
# IMAGE_PATH = BASE_PATH + QUERY_PATH
# artifacts = pd.read_csv("artifacts.csv")
# artifact_str = artifacts[artifacts["image_path"] == QUERY_PATH]["artifacts"].values.tolist()[0]
# objects_list = ast.literal_eval(artifact_str)
# objects = ". ".join([" ".join(obj.split("_")).lower() for obj in objects_list])
# objects += "."
# print(objects, type(objects))
