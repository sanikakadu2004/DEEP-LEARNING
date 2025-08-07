
# Multilayer Perceptron for Iris Flower Classification
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# 2. Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42)

# 3. Build the MLP model
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 4. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=5, validation_split=0.1, verbose=0)

# 6. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"
Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# 7. Predict and calculate manual accuracy
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

manual_accuracy = accuracy_score(y_true, y_pred)
print(f"Manual Accuracy: {manual_accuracy:.4f}")

# 8. Predict on a custom sample
sample = np.array([[6.1, 2.8, 4.7, 1.2]])  # Example flower
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
predicted_class = encoder.inverse_transform(prediction)
print(f"
Predicted Class: {predicted_class[0][0]} ({iris.target_names[predicted_class[0][0]]})")
