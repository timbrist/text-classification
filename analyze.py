import pandas as pd
from collections import Counter

# Read the CSV file (adjust the file name if needed)
df = pd.read_csv('data.csv')

# Initialize a Counter to store the counts for each cumulative hierarchy
hierarchy_counter = Counter()
df["Category"] = (
            df["Category"]
            .str.lower()
            .str.strip()
        )
# Process each row in the "Category" column
for row in df['Category']:
    # Split by '&' to account for multiple hierarchies in one row
    hierarchies = row.split('&')
    for hierarchy in hierarchies:
        # Split each hierarchy into levels using ':'
        levels = hierarchy.split(':')
        cumulative = []
        for level in levels:
            cumulative.append(level)
            # Join the cumulative levels with ':' to form the hierarchical key
            key = ":".join(cumulative)
            hierarchy_counter[key] += 1

# Convert the counter to a DataFrame for nicer output
df_hierarchy = pd.DataFrame(list(hierarchy_counter.items()), columns=['Hierarchy', 'Count'])
# Optional: sort by Hierarchy or by Count
df_hierarchy = df_hierarchy.sort_values(by='Hierarchy')

# Save the result to a new CSV file
df_hierarchy.to_csv('hierarchy_counts_testset.csv', index=False)

print("Hierarchy counts have been saved to 'hierarchy_counts_testset.csv'.")
