# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Preview the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Step 1: Data Cleaning
print("\nDataset Info:")
print(data.info())

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop('Cabin', axis=1, inplace=True)  # Dropping Cabin due to many missing values

# Check for duplicates
data.drop_duplicates(inplace=True)

# Step 2: Exploratory Data Analysis
# General statistics
print("\nStatistical Summary:")
print(data.describe())

# Survival rate analysis
survival_rate = data['Survived'].value_counts(normalize=True) * 100
print("\nSurvival Rate (%):")
print(survival_rate)

# Visualization: Survival by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=data, palette='viridis')
plt.title("Survival Count by Gender")
plt.show()

# Visualization: Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], kde=True, bins=30, color='blue')
plt.title("Age Distribution")
plt.show()

# Visualization: Survival by class
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=data, palette='coolwarm')
plt.title("Survival by Passenger Class")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
