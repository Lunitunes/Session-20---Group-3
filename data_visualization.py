import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
 
path = "processed_data/testingDataset.csv"
 
df = pd.read_csv(path)
 
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='encodedCategory', order=df['encodedCategory'].value_counts().index)
plt.title("Distribution of Traffic Categories")
plt.xticks(rotation=45)
plt.show()
 
fig, axes = plt.subplots(1, 3, figsize=(18,5))
 
sns.countplot(data=df, x='protocol_type', ax=axes[0])
axes[0].set_title("Protocol Type Distribution")
 
sns.countplot(data=df, x='service', ax=axes[1], order=df['service'].value_counts().iloc[:10].index)
axes[1].set_title("Top 10 Services")
 
sns.countplot(data=df, x='flag', ax=axes[2])
axes[2].set_title("Flag Distribution")
 
plt.tight_layout()
plt.show()
 
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()
 
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='category', y='src_bytes')
plt.ylim(0, 1000)
plt.title("Source Bytes by Category")
plt.show()