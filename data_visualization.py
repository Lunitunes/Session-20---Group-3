import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load your dataset
path = "raw_data/basic_data_4.csv"
df = pd.read_csv(path)

#Quick sanity check
print(df.head())
print(df.info())

#Label / Category distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="label", order=df["label"].value_counts().index)
plt.title("Distribution of Traffic Labels")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

#Protocol / Service / Flag distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.countplot(data=df, x="protocol_type", ax=axes[0])
axes[0].set_title("Protocol Type Distribution")

top_services = df["service"].value_counts().head(10).index
sns.countplot(data=df[df["service"].isin(top_services)], x="service", ax=axes[1])
axes[1].set_title("Top 10 Services")
axes[1].tick_params(axis="x", rotation=45)

sns.countplot(data=df, x="flag", ax=axes[2])
axes[2].set_title("Flag Distribution")

plt.tight_layout()
# plt.show()

#Correlation heatmap (numeric columns only)
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
# plt.show()

#Source Bytes by Label (clipped for readability)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="label", y=df["serror_rate"].clip(upper=1000))
plt.title("Serror Rate by Label (clipped at 1000)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
