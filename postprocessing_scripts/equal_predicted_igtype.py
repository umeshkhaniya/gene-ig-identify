import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the Excel file
df = pd.read_excel("merged1_result_uniquegenes_human.xlsx")  # replace with your actual file name

# Replace with your actual column names
col1 = 'ig_type'
col2 = 'predicted_igtype'
tm_col = 'score'

# Round TM-score
df['tm_score_rounded'] = df[tm_col].round(2)

# Filter range
df_filtered = df[(df['tm_score_rounded'] >= 0.4) & (df['tm_score_rounded'] <= 1)].copy()

# Create equality column
df_filtered['equal'] = df_filtered[col1] == df_filtered[col2]

# Create a sorted list of unique TM-score values in descending order
tm_bins = sorted(df_filtered['tm_score_rounded'].unique(), reverse=True)

# Prepare cumulative counts
equal_counts = []
total_counts = []

for threshold in tm_bins:
    subset = df_filtered[df_filtered['tm_score_rounded'] >= threshold]
    equal_counts.append(subset['equal'].sum())
    total_counts.append(len(subset))

# Build DataFrame for plotting
plot_df = pd.DataFrame({
    'TM-score ≥': tm_bins,
    'Cumulative Equal Count': equal_counts,
    'Cumulative Total Count': total_counts
})

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='TM-score ≥', y='Cumulative Equal Count', data=plot_df, label='Equal Count', color='blue', marker='o')
sns.lineplot(x='TM-score ≥', y='Cumulative Total Count', data=plot_df, label='Total Count', color='red', marker='s')
plt.xlabel("TM-score threshold (≥)")
plt.ylabel("Cumulative Count")
plt.title("Cumulative Equal vs Total Count by TM-score Threshold")
plt.grid(True)
plt.tight_layout()
plt.savefig("matching_igtype_comparision.png", dpi=300)
plt.show()
