# Titanic EDA Project in PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load the dataset
df = pd.read_csv('titanic.csv')

# 1. Explore data structure
print("Columns and data types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# 2. Identify trends and patterns
# Survival count
sns.countplot(x='Survived', data=df)
plt.title('Survival Counts')
plt.show()

# Survival by gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.show()

# Survival by class
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival by Class')
plt.show()

# Age distribution
plt.figure(figsize=(10,5))
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Age vs Survival
plt.figure(figsize=(10,5))
sns.kdeplot(df[df['Survived']==1]['Age'].dropna(), label='Survived', shade=True)
sns.kdeplot(df[df['Survived']==0]['Age'].dropna(), label='Did Not Survive', shade=True)
plt.title('Age Distribution by Survival')
plt.legend()
plt.show()

# 3. Test hypotheses
# Gender vs survival
contingency_table = pd.crosstab(df['Sex'], df['Survived'])
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f'Chi-square test for Gender vs Survival: p-value={p:.4f}')

# Class vs survival
contingency_table_class = pd.crosstab(df['Pclass'], df['Survived'])
chi2_class, p_class, _, _ = chi2_contingency(contingency_table_class)
print(f'Chi-square test for Class vs Survival: p-value={p_class:.4f}')

# 4. Detect data issues
print("\nMissing data percentage per column:")
print(df.isnull().mean() * 100)

# Fill missing age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")
