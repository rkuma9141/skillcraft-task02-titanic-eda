# TASK 02 - Titanic Dataset EDA
# Name: Raviranjan Kumar
# Internship: SkillCraft Technology
# Dataset: Titanic Dataset from Kaggle
# Description: This script performs data cleaning and exploratory data analysis (EDA)
#              to explore relationships, patterns, and trends in the Titanic dataset.

# Step 1: Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load Dataset
df = pd.read_csv("titanic.csv")  # Make sure the file is in the same folder or provide full path
print("First 5 rows of dataset:\n", df.head())

# Step 3: Data Info
print("\nDataset Info:")
print(df.info())

# Step 4: Handling Missing Values
print("\nMissing values before cleaning:\n", df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
print("\nMissing values after cleaning:\n", df.isnull().sum())

# Step 5: Basic Stats
print("\nStatistical Summary:\n", df.describe())

# Step 6: Exploratory Data Analysis (EDA)
# Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.clf()

# Survival by Gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival by Gender")
plt.savefig("survival_by_gender.png")
plt.clf()

# Age Distribution
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("age_distribution.png")
plt.clf()

# Step 7: Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.clf()

print("EDA Complete. Plots saved as images.")
