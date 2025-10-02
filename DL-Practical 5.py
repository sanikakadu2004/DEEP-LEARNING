
# Load the dataset
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
def load_csv(filepath):
    return pd.read_csv(filepath)
import pandas as pd
df = pd.read_csv('Breast Cancer Dataset.csv')
df.head()

# Check columns
print(df.columns)
# Check for missing values
print(df.info())

# Features = all gene expression columns (except target)
X = df.drop(columns=['metastasis'])
y = df['metastasis']

from sklearn.preprocessing import LabelEncoder, StandardScaler
# Encode tumor/normal as 0/1
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['FPKM']])

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape to 3D: (samples, timesteps, features)
X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

# Build RNN
model = Sequential()
model.add(SimpleRNN(32, activation='tanh', input_shape=(X_train_rnn.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the Model
history = model.fit(
    X_train_rnn, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=256,
    verbose=1
)

loss, accuracy = model.evaluate(X_test_rnn, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Predictions
y_pred = model.predict(X_test_rnn)
y_pred_classes = (y_pred > 0.5).astype(int)
print("Sample Predictions:")
for i in range(5):
    print(f"Actual: {y_test[i]}, Predicted: {y_pred_classes[i][0]}")
import matplotlib.pyplot as plt
# Create side-by-side plots
plt.figure(figsize=(12,5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Training vs Validation Loss")

plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal','Tumor'], yticklabels=['Normal','Tumor'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['Normal','Tumor']))

from sklearn.metrics import roc_curve, auc
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
