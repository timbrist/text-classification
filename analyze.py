

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
df = pd.read_csv('data.csv')

# Step 2: Initialize a Counter to store counts for each cumulative hierarchy
hierarchy_counter = Counter()

df["Category"] = (
            df["Category"]
            .str.lower()
            .str.strip()
        )
# Process each row in the "Category" column
for row in df['Category']:
    # Split the row by '&' to handle multiple hierarchies in one row
    hierarchies = row.split('&')
    for hierarchy in hierarchies:
        # Split the hierarchy into levels using ':'
        levels = hierarchy.split(':')
        cumulative = []
        for level in levels:
            cumulative.append(level)
            # Build the cumulative key (e.g., "A", "A:B", "A:B:C")
            key = ":".join(cumulative)
            hierarchy_counter[key] += 1

# Step 3: Calculate total count of all cumulative keys
total_count = sum(hierarchy_counter.values())

# Step 4: Prepare data with counts and percentages
data = []
for key, count in hierarchy_counter.items():
    percentage = (count / total_count) * 100
    data.append((key, count, percentage))

# Create a DataFrame for nicer output
df_hierarchy = pd.DataFrame(data, columns=['Hierarchy', 'Count', 'Percentage'])
df_hierarchy = df_hierarchy.sort_values(by='Hierarchy')

# Save the results to a CSV file
df_hierarchy.to_csv('hierarchy_counts_with_percentages.csv', index=False)
print("Hierarchy counts with percentages have been saved to 'hierarchy_counts_with_percentages.csv'.")

# Step 5: Create a pie chart for first-level categories (those without a colon)
# These represent the highest level (e.g., "A" if your data always starts with "A")
first_level_counts = {k: v for k, v in hierarchy_counter.items() if ':' not in k}

plt.figure(figsize=(8, 8))
plt.pie(first_level_counts.values(), labels=first_level_counts.keys(), autopct='%1.1f%%', startangle=140)
plt.title('Distribution of First-Level Categories')
plt.axis('equal')  # Ensure the pie chart is circular
plt.show()
