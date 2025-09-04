
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
diabetes = load_diabetes()
X = diabetes.data[:, np.newaxis, 2]   # BMI feature (3rd column)
y = diabetes.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add intercept term (bias)
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Step 2: Gradient Descent Implementation
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

def gradient_descent(X, y, theta, learning_rate=0.1, iterations=1000):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        gradients = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradients
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

# Initialize parameters
theta_init = np.zeros((2, 1))
y_train_reshaped = y_train.reshape(-1, 1)

# Run Gradient Descent
final_theta, cost_history = gradient_descent(X_train_b, y_train_reshaped, theta_init, learning_rate=0.1, iterations=1000)

# Step 3: Normal Equation (Closed-Form Solution)
normal_theta = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train_reshaped)

# Step 4: Evaluation
y_pred_gd = X_test_b.dot(final_theta)
y_pred_ne = X_test_b.dot(normal_theta)

mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)

mse_ne = mean_squared_error(y_test, y_pred_ne)
r2_ne = r2_score(y_test, y_pred_ne)

print("Gradient Descent Parameters:", final_theta.ravel())
print("Normal Equation Parameters:", normal_theta.ravel())
print("
Performance on Test Set:")
print(f"Gradient Descent -> MSE: {mse_gd:.4f}, R²: {r2_gd:.4f}")
print(f"Normal Equation -> MSE: {mse_ne:.4f}, R²: {r2_ne:.4f}")

# Step 5: Visualization
plt.figure(figsize=(12,5))

# Cost vs Iterations
plt.subplot(1,2,1)
plt.plot(range(1, len(cost_history)+1), cost_history, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Convergence of Gradient Descent')

# Regression Line
plt.subplot(1,2,2)
plt.scatter(X_test, y_test, color='gray', alpha=0.6, label="Test Data")
plt.plot(X_test, y_pred_gd, color='red', label="Gradient Descent")
plt.plot(X_test, y_pred_ne, color='green', linestyle='dashed', label="Normal Equation")
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title('Regression Line Comparison')
plt.legend()

plt.tight_layout()
plt.show()
