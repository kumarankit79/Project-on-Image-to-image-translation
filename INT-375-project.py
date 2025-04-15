# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the dataset
df =pd.read_csv("/content/drive/MyDrive/Sketch Image to Image Translation Data.csv")
df.head()


# Overview of Dataset
df.info()

# Statistical information of numeric features of the dataset

print(df.describe())

# Count of unique values for each categorical columns

categorical_cols = [
    'Sketch_Artist_ID', 'Object_ID', 'Sketch_type', 'Sketch_Name', 'Real_Image_Name',
    'category of sketch', 'sub-category of sketch', 'Difficulty_Level'
]

for col in categorical_cols:
    print(f"For column '{col}', there are {df[col].nunique()} unique values.")


#Scatter plot for Sketch Width vs Sketch Height

df_top50 = df.head(50)

plt.figure(figsize=(8, 4))
plt.scatter(df_top50['Sketch_width'], df_top50['Sketch_height'], color='purple')
plt.xlabel('Sketch Width')
plt.ylabel('Sketch Height')
plt.title('Scatter Plot for Sketch Width vs Sketch Height')
plt.tight_layout()
plt.show()


# Histogram

df_top50 = df.head(50)

for col in ['matching_score', 'Human_Evaluation_Score']:
    p = sns.histplot(df_top50[col], bins=10, color='skyblue')
    for bar in p.patches:
        h = bar.get_height()
        if h > 0: p.text(bar.get_x() + bar.get_width()/2, h + 0.2, int(h), ha='center', fontsize=8)
    plt.title(f"Distribution of {col} in Histogram ")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# Pie -Chart

col = 'sketch_complexity'  # You can change this to any categorical column
counts = df[col].value_counts()

plt.figure(figsize=(4, 4))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'pink', 'lightgreen', 'orange', 'violet'])
plt.title(f"{col} Distribution in pie Chart")
plt.axis('equal')
plt.show()


# Boxplot

cols = ['matching_score']

for col in cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Correlation bewtween Sketch_width, Sketch_height, Real_image_Width & Real_image_height.

cols = ['Sketch_width', 'Sketch_height', 'Real_image_Width', 'Real_image_height']

correlation_matrix = df[cols].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Between Sketch & Real Image Dimensions")
plt.tight_layout()
plt.show()


# Bar-plot
columns = ['Sketch_width']
df_top50 = df.head(20)

for col in columns:
    plt.figure(figsize=(10, 4))
    plt.bar(df_top50.index, df_top50[col], color='skyblue')
    plt.title(f"{col} distribution in Bar-Plot")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# Line-Plot
columns = ['Sketch_width']
df_top50 = df.head(30)

for col in columns:
    plt.figure(figsize=(10, 4))
    plt.plot(df_top50.index, df_top50[col], marker='o', linestyle='-', color='teal')
    plt.title(f"{col} - Line Plot (Top 30)")
    plt.xlabel("Index")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# Line-chart

df_top50 = df.head(30)

plt.figure(figsize=(12, 6))
plt.plot(df_top50['Sketch_width'], label='Sketch Width', marker='o')


plt.title("Line Chart of Sketch and Real Image Dimensions (Top 30 Rows)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Z-Test

from scipy.stats import zscore

columns = ['Sketch_width', 'Sketch_height', 'Real_image_Width', 'Real_image_height']
z_scores = df[columns].apply(zscore)

print(z_scores.head())