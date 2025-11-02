"""
Introduction to Machine Learning Basics
========================================

This script introduces fundamental concepts in machine learning.
"""

import numpy as np
import matplotlib.pyplot as plt

# What is Machine Learning?
print("=" * 60)
print("What is Machine Learning?")
print("=" * 60)
print("""
Machine Learning is a subset of artificial intelligence that enables
systems to learn and improve from experience without being explicitly
programmed.

Main Types of Machine Learning:
1. Supervised Learning - Learning from labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through trial and error
""")

# Simple example: Understanding data
print("\n" + "=" * 60)
print("Example: Working with Data")
print("=" * 60)

# Create sample data
np.random.seed(42)
data = np.random.randn(100, 2)

print(f"\nGenerated {len(data)} data points with 2 features")
print(f"Shape of data: {data.shape}")
print(f"\nFirst 5 data points:")
print(data[:5])

print(f"\nBasic statistics:")
print(f"Mean: {data.mean(axis=0)}")
print(f"Standard deviation: {data.std(axis=0)}")

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Sample Data Visualization')
plt.grid(True, alpha=0.3)
plt.savefig('examples/01_introduction/data_visualization.png')
print("\nVisualization saved to 'data_visualization.png'")

print("\n" + "=" * 60)
print("Key Concepts in Machine Learning:")
print("=" * 60)
print("""
1. Features (X): Input variables used to make predictions
2. Target (y): Output variable we want to predict
3. Training: Process of learning from data
4. Testing: Evaluating model performance on unseen data
5. Model: Mathematical representation learned from data
""")
