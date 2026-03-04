import pandas as pd
import matplotlib.pyplot as plt

# Load the classified results file
file_path = "classified_results.csv"
df = pd.read_csv(file_path)

df["Category"] = (
            df["Category"]
            .str.lower()
            .str.strip()
            .apply(lambda x: [label.strip() for label in x.split("&")])
        )


# label_counts = df.iloc[:, 1].value_counts()  # Assuming second column has the labels
# print(label_counts)
# # Plot pie chart
# plt.figure(figsize=(10, 10))
# plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
# plt.title("Distribution of Classified Traffic Scenarios")
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# # Show the pie chart
# plt.show()