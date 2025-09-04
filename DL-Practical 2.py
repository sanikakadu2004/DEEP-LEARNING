
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Generate synthetic dataset
# Fix random seed for reproducibility
np.random.seed(42)
# Generate 100 points between 0 and 10
X = np.linspace(0, 10, 100)
# True relationship is sine wave
y_true = np.sin(X)
# Add Gaussian noise
y = y_true + np.random.normal(scale=0.3, size=len(X))
# Reshape X into column vector for sklearn
X = X.reshape(-1, 1)

# Split dataset into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Train polynomial regression models of varying degree
degrees = [1, 4, 15]  # Low, medium, high complexity
plt.figure(figsize=(15, 5))

for i, d in enumerate(degrees, 1):
    # Polynomial features
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predictions on train and test sets
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # Calculate training and test errors (MSE)
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    # Plot
    plt.subplot(1, 3, i)
    plt.scatter(X_train, y_train, color="blue", label="Training Data")
    plt.scatter(X_test, y_test, color="red", label="Testing Data")
    
    # Smooth curve for visualization
    X_range = np.linspace(0, 10, 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)
    plt.plot(X_range, y_range_pred, color="green", linewidth=2, label=f"Degree {d}")
    
    plt.title(f"Degree {d}
Train Error={train_error:.3f}, Test Error={test_error:.3f}")
    plt.legend()

plt.tight_layout()
plt.show()
