import pandas as pd
import json

# Load the CSV file
file_path = "cleaned_data.csv"
df = pd.read_csv(file_path)

# Initialize a new dictionary to store categorized IDs
category_dict_split = {}

# Process each row in the dataframe
for _, row in df.iterrows():
    categories = row["Category"].split("&")  # Split categories by "&"
    id_value = str(row["id"])  # Convert ID to string

    for category in categories:
        category = category.strip()  # Remove any extra whitespace
        
        if category not in category_dict_split:
            category_dict_split[category] = {"id": []}
        
        category_dict_split[category]["id"].append(id_value)

# Save the updated result as a JSON file
output_file_path_split = "sorted_categories_split.json"

with open(output_file_path_split, "w") as json_file:
    json.dump(category_dict_split, json_file, indent=4)

# Output the file path
print(output_file_path_split)
