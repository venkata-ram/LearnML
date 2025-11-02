"""
Classification with Logistic Regression
========================================

Classification is used to predict categorical outcomes.
This example uses logistic regression for binary classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

# Generate sample classification data
print("=" * 60)
print("Binary Classification Example")
print("=" * 60)

np.random.seed(42)
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

print(f"\nGenerated {len(X)} samples for classification")
print(f"Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution: {np.bincount(y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.2%}")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize decision boundary
plt.figure(figsize=(12, 5))

# Plot 1: Training data
plt.subplot(1, 2, 1)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], 
            alpha=0.6, label='Class 0', color='blue')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], 
            alpha=0.6, label='Class 1', color='red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Test data with predictions
plt.subplot(1, 2, 2)
correct = y_test == y_pred
plt.scatter(X_test[correct][:, 0], X_test[correct][:, 1], 
            alpha=0.6, label='Correct', color='green', marker='o')
plt.scatter(X_test[~correct][:, 0], X_test[~correct][:, 1], 
            alpha=0.6, label='Incorrect', color='red', marker='x', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/03_classification/classification.png')
print("\nVisualization saved to 'classification.png'")
