
!pip install -q imbalanced-learn

# 1. Import libraries
# Imports and reproducibility
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)

from imblearn.over_sampling import SMOTE  # optional
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# 2. Load dataset and basic EDA
# Upload file in Colab
from google.colab import files
uploaded = files.upload()
# Load
df = pd.read_csv('creditcard.csv')
print("Shape:", df.shape)
print(df['Class'].value_counts())
df.head()

# Quick class distribution
sns.countplot(x='Class', data=df)
plt.title('Class distribution (0 = normal, 1 = fraud)')
plt.show()

# Summary stats
print(df.describe())
# Note: Features V1..V28 are PCA components from original dataset, Time and Amount are raw.

# 3. Preprocessing (scale Time & Amount only, avoid leakage)
# Split first to avoid data leakage when scaling
X = df.drop(columns=['Class'])
y = df['Class']

# train/test split (stratify to keep imbalance ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y)

# Scale ONLY Time and Amount (other features V1..V28 are already PCA-like)
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Fraud counts (train/test):", y_train.sum(), y_test.sum())


# For autoencoder training: use only NORMAL transactions
X_train_ae = X_train_full[y_train_full == 0]
print("AE training samples (normal only):", X_train_ae.shape)

# 4. Train Autoencoder on normal transactions only
# Prepare only normal transactions for autoencoder
X_train_ae = X_train[y_train == 0].values  # numpy array
input_dim = X_train_ae.shape[1]
encoding_dim = 16  # tuneable

# Build autoencoder
def build_autoencoder(input_dim, encoding_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu', name='encoded')(x)
    x = layers.Dense(32, activation='relu')(encoded)
    x = layers.Dense(64, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    ae = models.Model(inputs=inp, outputs=decoded, name='autoencoder')
    encoder = models.Model(inputs=inp, outputs=encoded, name='encoder')
    return ae, encoder

ae, encoder = build_autoencoder(input_dim, encoding_dim)
ae.compile(optimizer=optimizers.Adam(1e-3), loss='mse')
ae.summary()

# 5. Train AE (with early stopping) and visualize loss
# Train/validation split for AE
from sklearn.model_selection import train_test_split
X_ae_tr, X_ae_val = train_test_split(X_train_ae, test_size=0.1, random_state=seed)

es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = ae.fit(
    X_ae_tr, X_ae_tr,
    epochs=25,
    batch_size=2048,
    validation_data=(X_ae_val, X_ae_val),
    callbacks=[es],
    verbose=2
)

# Plot AE loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.legend()
plt.title('Autoencoder training loss')
plt.show()

# 6. Compute per-sample reconstruction error (train & test)
# Predict reconstructions for train and test
X_train_vals = X_train.values
X_test_vals = X_test.values

train_recon = ae.predict(X_train_vals)
test_recon = ae.predict(X_test_vals)

# Per-sample MSE
train_mse = np.mean(np.power(X_train_vals - train_recon, 2), axis=1)
test_mse = np.mean(np.power(X_test_vals - test_recon, 2), axis=1)

# Attach AE error as a column for further use
X_train_with_ae = X_train.copy()
X_test_with_ae = X_test.copy()
X_train_with_ae['AE_Error'] = train_mse
X_test_with_ae['AE_Error'] = test_mse

# Quick visualization: MSE distribution (test)
plt.figure(figsize=(8,4))
sns.histplot(test_mse[y_test==0], bins=200, label='normal', stat='density', alpha=0.6)
sns.histplot(test_mse[y_test==1], bins=200, label='fraud', stat='density', alpha=0.6)
plt.yscale('log')
plt.legend()
plt.title('Reconstruction error (test) - normal vs fraud')
plt.show()

# 7. Simple AE thresholding (baseline) — choose threshold from train normal percentiles
# Choose threshold using train normal MSE percentiles (avoid using test labels)
threshold = np.percentile(train_mse[y_train==0], 99.5)  # tuneable percentile
print("Threshold (99.5 percentile of train normal):", threshold)

pred_ae_test = (test_mse > threshold).astype(int)

print("AE-only baseline results")
print("
 Confusion matrix (AE):")
print(confusion_matrix(y_test, pred_ae_test))

print("
 Classification Report (AE):")
print(classification_report(y_test, pred_ae_test, digits=4))

print("ROC-AUC (AE error):", roc_auc_score(y_test, test_mse))
print("PR-AUC (AE error):", average_precision_score(y_test, test_mse))

# Visualization
cm_ae = confusion_matrix(y_test, pred_ae_test)
fpr, tpr, _ = roc_curve(y_test, test_mse)
precision, recall, _ = precision_recall_curve(y_test, test_mse)
auc = roc_auc_score(y_test, test_mse)
pr_auc = average_precision_score(y_test, test_mse)

# Create subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix
sns.heatmap(cm_ae, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix - Autoencoder')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# ROC Curve
axes[1].plot(fpr, tpr, label=f"AUC = {auc:.4f}", color='blue')
axes[1].plot([0,1], [0,1], '--', color='grey')
axes[1].set_title('ROC Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

# Precision-Recall Curve
axes[2].plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}", color='green')
axes[2].set_title('Precision-Recall Curve')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].legend()

plt.tight_layout()
plt.show()

# 8. Build hybrid features: encoded vector + AE_Error
# Get encoded representations for train & test
encoded_train = encoder.predict(X_train_vals)  # shape (n_train, encoding_dim)
encoded_test  = encoder.predict(X_test_vals)

# Combine encoded + AE_Error as features for MLP
X_clf_train = np.hstack([encoded_train, train_mse.reshape(-1,1)])
X_clf_test  = np.hstack([encoded_test,  test_mse.reshape(-1,1)])
y_clf_train = y_train.values
y_clf_test  = y_test.values

print("Classifier features shape:", X_clf_train.shape, X_clf_test.shape)
print("Fraud counts in clf train/test:", y_clf_train.sum(), y_clf_test.sum())

# 9. Handle class imbalance (two common options)
# Option A: use class weights (fast, recommended)
from sklearn.utils import class_weight
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_clf_train), y=y_clf_train)
class_weights = {0: weights[0], 1: weights[1]}
print("Class weights:", class_weights)

# Option B: SMOTE over-sampling (uncomment to use). Works on numeric encoded features.
print('Before SMOTE:', np.bincount(y_clf_train))
sm = SMOTE(random_state=seed)
X_train_bal, y_train_bal = sm.fit_resample(X_clf_train, y_clf_train)
print('After SMOTE:', np.bincount(y_train_bal))

# 10. Build and train MLP classifier (on hybrid features)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_mlp(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    return model

mlp = build_mlp(X_clf_train.shape[1])
mlp.summary()

# Use either original or SMOTE-resampled training set
# X_train_to_use, y_train_to_use = (X_clf_train_res, y_clf_train_res) if using SMOTE else (X_clf_train, y_clf_train)
X_train_to_use, y_train_to_use = X_clf_train, y_clf_train

es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_mlp = mlp.fit(
    X_train_to_use, y_train_to_use,
    validation_split=0.1,
    epochs=25,
    batch_size=1024,
    class_weight=class_weights,  # if using SMOTE, remove class_weight
    callbacks=[es],
    verbose=2
)

# Plot MLP training AUC/loss
plt.plot(history_mlp.history['loss'], label='train_loss')
plt.plot(history_mlp.history['val_loss'], label='val_loss')
plt.legend(); plt.title('MLP loss'); plt.show()

# 11. Evaluate MLP on test set (metrics + curves)
y_proba = mlp.predict(X_clf_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)
print("MLP results")
print("
 Confusion matrix (MLP):")
print(confusion_matrix(y_clf_test, y_pred))

print("
 Classification Report (MLP):")
print(classification_report(y_clf_test, y_pred, digits=4))
print("ROC-AUC (MLP):", roc_auc_score(y_clf_test, y_proba))
print("PR-AUC (MLP):", average_precision_score(y_clf_test, y_proba))

# Visualization
cm_mlp = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_clf_test, y_proba)
precision, recall, _ = precision_recall_curve(y_clf_test, y_proba)
auc = roc_auc_score(y_clf_test, y_proba)
pr_auc = average_precision_score(y_clf_test, y_proba)

# Create subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', ax=axes[0])
axes[0].set_title('Confusion Matrix - MLP')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# ROC Curve
axes[1].plot(fpr, tpr, label=f"AUC = {auc:.4f}", color='darkgreen')
axes[1].plot([0,1], [0,1], '--', color='grey')
axes[1].set_title('ROC Curve - MLP')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

# Precision-Recall Curve
axes[2].plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}", color='limegreen')
axes[2].set_title('Precision-Recall Curve - MLP')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].legend()

plt.tight_layout()
plt.show()

# 12. Save models and scalers
# Save Keras models
ae.save("autoencoder_model.h5")
encoder.save("encoder_model.h5")
mlp.save("mlp_hybrid_model.h5")

# Save scaler and other artifacts with pickle
import pickle
with open("scaler_time_amount.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved models and scaler.")

# 13. Comparison and Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Generate confusion matrices of AE & MLP
cm_ae = confusion_matrix(y_test, pred_ae_test)
cm_mlp = confusion_matrix(y_clf_test, y_pred)

# Generate classification reports as DataFrames for cleaner display AE & MLP
report_ae = pd.DataFrame(classification_report(y_test, pred_ae_test, digits=4, output_dict=True)).transpose()
report_mlp = pd.DataFrame(classification_report(y_clf_test, y_pred, digits=4, output_dict=True)).transpose()

# Print both reports side by side
print("
===== Model Performance Comparison (Text Output) =====")
print("
AUTOENCODER (AE) CLASSIFICATION REPORT:")
print(report_ae)

print("
MLP CLASSIFICATION REPORT:")
print(report_mlp)

# Plot both confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_ae, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix - Autoencoder (AE)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Confusion Matrix - Hybrid MLP')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 14. Summary
# Model Performance Summary
ae_accuracy = accuracy_score(y_test, pred_ae_test)
ae_precision = precision_score(y_test, pred_ae_test)
ae_recall = recall_score(y_test, pred_ae_test)
ae_f1 = f1_score(y_test, pred_ae_test)

mlp_accuracy = accuracy_score(y_test, y_pred)
mlp_precision = precision_score(y_test, y_pred)
mlp_recall = recall_score(y_test, y_pred)
mlp_f1 = f1_score(y_test, y_pred)

# Create summary DataFrame
summary = pd.DataFrame({
    'Model': ['Autoencoder', 'MLP'],
    'Accuracy': [ae_accuracy, mlp_accuracy],
    'Precision': [ae_precision, mlp_precision],
    'Recall': [ae_recall, mlp_recall],
    'F1-Score': [ae_f1, mlp_f1]
})

print("
 Model Performance Summary:")
print(summary)

# Visualization of Comparison
plt.figure(figsize=(8,5))
summary.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar')
plt.title('Autoencoder vs MLP - Performance Comparison')
plt.ylabel('Score')
plt.ylim(0,1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Short Summary
print("
 Short Analysis Summary:")
print(f"Autoencoder achieved {ae_accuracy:.2f} accuracy by reconstructing normal transactions and identifying high-error ones as frauds.")
print(f"MLP achieved {mlp_accuracy:.2f} accuracy using supervised learning with class balancing.")
if mlp_accuracy > ae_accuracy:
    print("MLP performed better overall due to learning from both normal and fraudulent samples.")
else:
    print("Autoencoder detected anomalies effectively, showing strong unsupervised detection capability.")
