"""
Linear Regression - Supervised Learning Example
================================================

Linear regression is used to predict a continuous output variable
based on one or more input features.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
print("=" * 60)
print("Linear Regression Example")
print("=" * 60)

np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
y = 2 * X + 1 + np.random.randn(100, 1) * 2  # y = 2x + 1 + noise

print(f"\nGenerated {len(X)} samples for regression")
print(f"True relationship: y = 2x + 1 (with noise)")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nModel learned:")
print(f"Coefficient (slope): {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
plt.scatter(X_test, y_test, alpha=0.6, label='Testing data', color='orange')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prediction line')
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Linear Regression Example')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('examples/02_regression/linear_regression.png')
print("\nVisualization saved to 'linear_regression.png'")
