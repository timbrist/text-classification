import pandas as pd

# Load the CSV file
file_path = "cleaned_data.csv"
df = pd.read_csv(file_path)

# Function to clean and format the Category column
def clean_category(category):
    categories = category.lower().split("&")  # Convert to lowercase and split by "&"
    cleaned_categories = [":".join(cat.strip().split(":")) for cat in categories]  # Remove spaces and format
    return "&".join(cleaned_categories)  # Join back with "&"

# Apply cleaning function to the Category column
df["Category"] = df["Category"].apply(clean_category)

# Save the cleaned data to a new CSV file
cleaned_file_path = "cleaned_data.csv"
df.to_csv(cleaned_file_path, index=False)

# Output the file path
print(cleaned_file_path)