import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('percent_aligned.tsv', sep='\t')
# Remove % from the string and convert to float
meta_df = pd.read_csv('metadata-Yasmin_FMT.tsv', sep='\t')
# join the two dataframes
df = df.join(meta_df.set_index('sample'), on='sample')
# df has 'group' category. Groupby this category to show boxplots
sns.boxplot(data=df, x='group', y='% Aligned')
# Show also the data points
sns.swarmplot(data=df, x='group', y='% Aligned', color='black')
plt.show()
# Plot also the 'M Aligned' column
sns.boxplot(data=df, x='group', y='M Aligned')
sns.swarmplot(data=df, x='group', y='M Aligned', color='black')
plt.show()
df['Total Reads'] = df['M Aligned'] / df['% Aligned'] * 100
sns.boxplot(data=df, x='group', y='Total Reads')
sns.swarmplot(data=df, x='group', y='Total Reads', color='black')
plt.show()
