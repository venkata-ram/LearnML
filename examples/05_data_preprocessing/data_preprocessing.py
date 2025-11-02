"""
Data Preprocessing - Essential Steps for ML
============================================

Data preprocessing is crucial for building effective ML models.
This example demonstrates common preprocessing techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

print("=" * 60)
print("Data Preprocessing Techniques")
print("=" * 60)

# Create sample dataset with common data issues
np.random.seed(42)
data = {
    'age': [25, 30, np.nan, 35, 40, 22, 28, np.nan, 45, 33],
    'salary': [50000, 60000, 55000, 80000, np.nan, 45000, 62000, 70000, 90000, 58000],
    'department': ['Sales', 'IT', 'HR', 'Sales', 'IT', 'HR', 'Sales', 'IT', 'HR', 'Sales'],
    'rating': [3.5, 4.0, 2.5, 4.5, 3.0, 2.0, 4.2, 3.8, 4.8, 3.6]
}

df = pd.DataFrame(data)

print("\nOriginal Data:")
print(df)
print(f"\nMissing values:\n{df.isnull().sum()}")

# 1. Handling Missing Values
print("\n" + "=" * 60)
print("1. Handling Missing Values")
print("=" * 60)

# Impute numerical columns with mean
imputer = SimpleImputer(strategy='mean')
df_imputed = df.copy()
df_imputed[['age', 'salary']] = imputer.fit_transform(df[['age', 'salary']])

print("\nAfter imputation:")
print(df_imputed)

# 2. Feature Scaling
print("\n" + "=" * 60)
print("2. Feature Scaling")
print("=" * 60)

# Standardization (z-score normalization)
scaler_standard = StandardScaler()
df_standardized = df_imputed.copy()
df_standardized[['age', 'salary']] = scaler_standard.fit_transform(
    df_imputed[['age', 'salary']]
)

print("\nStandardized data (mean=0, std=1):")
print(df_standardized[['age', 'salary']])

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
df_normalized = df_imputed.copy()
df_normalized[['age', 'salary']] = scaler_minmax.fit_transform(
    df_imputed[['age', 'salary']]
)

print("\nNormalized data (range 0-1):")
print(df_normalized[['age', 'salary']])

# 3. Encoding Categorical Variables
print("\n" + "=" * 60)
print("3. Encoding Categorical Variables")
print("=" * 60)

# Label Encoding
label_encoder = LabelEncoder()
df_encoded = df_imputed.copy()
df_encoded['department_encoded'] = label_encoder.fit_transform(df_imputed['department'])

print("\nLabel Encoded departments:")
print(df_encoded[['department', 'department_encoded']])

# One-Hot Encoding
df_onehot = pd.get_dummies(df_imputed, columns=['department'], prefix='dept')
print("\nOne-Hot Encoded:")
print(df_onehot)

# 4. Visualize preprocessing effects
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original data
axes[0, 0].scatter(df_imputed['age'], df_imputed['salary'], alpha=0.6)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Salary')
axes[0, 0].set_title('Original Data')
axes[0, 0].grid(True, alpha=0.3)

# Standardized data
axes[0, 1].scatter(df_standardized['age'], df_standardized['salary'], alpha=0.6, color='orange')
axes[0, 1].set_xlabel('Age (Standardized)')
axes[0, 1].set_ylabel('Salary (Standardized)')
axes[0, 1].set_title('Standardized Data')
axes[0, 1].grid(True, alpha=0.3)

# Normalized data
axes[1, 0].scatter(df_normalized['age'], df_normalized['salary'], alpha=0.6, color='green')
axes[1, 0].set_xlabel('Age (Normalized)')
axes[1, 0].set_ylabel('Salary (Normalized)')
axes[1, 0].set_title('Normalized Data (0-1)')
axes[1, 0].grid(True, alpha=0.3)

# Distribution comparison
axes[1, 1].hist(df_imputed['salary'], bins=10, alpha=0.5, label='Original', color='blue')
axes[1, 1].hist(df_standardized['salary'] * df_imputed['salary'].std() + df_imputed['salary'].mean(), 
                bins=10, alpha=0.5, label='After Standardization', color='orange')
axes[1, 1].set_xlabel('Salary')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/05_data_preprocessing/preprocessing.png')
print("\nVisualization saved to 'preprocessing.png'")

print("\n" + "=" * 60)
print("Key Preprocessing Techniques Summary:")
print("=" * 60)
print("""
1. Missing Value Handling:
   - Mean/Median/Mode imputation
   - Forward/Backward fill
   - Drop missing values

2. Feature Scaling:
   - Standardization (Z-score)
   - Min-Max Normalization
   - Robust Scaling

3. Encoding:
   - Label Encoding (ordinal)
   - One-Hot Encoding (nominal)
   - Target Encoding

4. Feature Engineering:
   - Creating new features
   - Polynomial features
   - Feature selection
""")
