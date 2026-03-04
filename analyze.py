import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
df = pd.read_csv('classified_results.csv')
df["Category"] = (
            df["Category"]
            .str.lower()
            .str.strip()
        )
# Step 2: Initialize a Counter to store counts for each cumulative hierarchy
hierarchy_counter = Counter()

# Process each row in the "Category" column
for row in df['Category']:
    # Split by '&' to handle multiple hierarchies in one row
    hierarchies = row.split('&')
    for hierarchy in hierarchies:
        # Split the hierarchy into levels using ':'
        levels = hierarchy.split(':')
        cumulative = []
        for level in levels:
            cumulative.append(level.strip())
            # Build the cumulative key (e.g., "A", "A:B", "A:B:C")
            key = ":".join(cumulative)
            hierarchy_counter[key] += 1

# Step 3: Calculate total count of all cumulative keys
total_count = len(df["Category"])

print(total_count)
# Step 4: Prepare data with counts and percentages
data = []
for key, count in hierarchy_counter.items():
    percentage = (count / total_count)
    data.append((key, count, percentage))

# Extract first-level data (keys with no colon) using the computed percentages
first_level_percentages = { key: percentage for key, count, percentage in data if ':' not in key }


# Create a DataFrame for nicer output and save to a CSV file
df_hierarchy = pd.DataFrame(data, columns=['Hierarchy', 'Count', 'Percentage'])
df_hierarchy = df_hierarchy.sort_values(by='Hierarchy')
df_hierarchy.to_csv('hierarchy_counts_with_percentages.csv', index=False)
print("Hierarchy counts with percentages have been saved to 'hierarchy_counts_with_percentages.csv'.")

# Step 5: Create a pie chart for first-level categories (those without a colon)
first_level_counts = {k: v for k, v in hierarchy_counter.items() if ':' not in k}
print(first_level_counts)
# Define a color mapping for specific labels. For example, "road condition" will always be blue.
color_mapping = {
    "road condition": "cornflowerblue",
    "traffic conditions": "darkorange",
    "interaction with other road users": "violet",
    "ego car driving intention":"skyblue",
    "environment":"limegreen",
    "unclassified":"peru",
}

# Generate a list of colors for the pie chart, using the mapping if available, or a default color cycle.
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = []
for i, label in enumerate(first_level_counts.keys()):
    # Lowercase comparison for flexibility (if your labels may vary in case)
    if label.lower() in color_mapping:
        colors.append(color_mapping[label.lower()])
    else:
        colors.append(default_colors[i % len(default_colors)])

print(first_level_percentages.values())

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(
   list(first_level_percentages.values()),
    labels=list(first_level_percentages.keys()),
        autopct='%1.2f%%',
        startangle=140,
        colors=colors)
plt.title('Distribution of First-Level Categories')
plt.axis('equal')  # Ensure the pie chart is circular
plt.show()
