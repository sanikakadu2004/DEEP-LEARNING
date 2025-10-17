
pip install tensorflow matplotlib scikit-learn

# 1. Imports & seed
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools

# reproducibility
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

from tensorflow.keras import layers, models, callbacks, utils
from sklearn.metrics import confusion_matrix, classification_report

# 2. Load CIFAR-10 and create a mini subset
# Load CIFAR-10 (50k train, 10k test)
(x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()
y_train_full = y_train_full.flatten()
y_test_full = y_test_full.flatten()

# Choose a subset of classes (e.g., classes 0..4 => airplane, automobile, bird, cat, deer)
selected_classes = [0,1,2,3,4]
n_per_class_train = 800   # number of train images per class (adjust for "mini")
n_per_class_val = 100
n_per_class_test = 100

def make_subset(x, y, classes, per_class):
    idxs = []
    for c in classes:
        c_idxs = np.where(y == c)[0][:per_class]
        # If there aren't enough in the first slice, take more from the array
        if len(c_idxs) < per_class:
            c_idxs = np.where(y == c)[0][:per_class]
        idxs.extend(list(c_idxs))
    return x[idxs], y[idxs]

# Build train, val, test subsets
# First shuffle full sets to avoid always picking same images
perm_train = np.random.permutation(len(x_train_full))
x_train_shuf = x_train_full[perm_train]
y_train_shuf = y_train_full[perm_train]

perm_test = np.random.permutation(len(x_test_full))
x_test_shuf = x_test_full[perm_test]
y_test_shuf = y_test_full[perm_test]

# Build train and val from train_shuf
x_train_sub, y_train_sub = make_subset(x_train_shuf, y_train_shuf, selected_classes, n_per_class_train)
# split a validation set out of train_sub
val_split = 0.2
num_val = int(len(x_train_sub) * val_split)
x_val = x_train_sub[:num_val]
y_val = y_train_sub[:num_val]
x_train = x_train_sub[num_val:]
y_train = y_train_sub[num_val:]

# Build test subset from test_shuf
x_test, y_test = make_subset(x_test_shuf, y_test_shuf, selected_classes, n_per_class_test)

print("Train:", x_train.shape, y_train.shape)
print("Val:  ", x_val.shape, y_val.shape)
print("Test: ", x_test.shape, y_test.shape)

# Visualize the mini dataset samples
# Class names for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Display a few sample images from the training subset
plt.figure(figsize=(10, 6))
for i in range(12):  # show first 12 images
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_train[i])
    label = y_train[i]
    plt.title(class_names[label])
    plt.axis('off')
plt.suptitle("Sample Images from Mini CIFAR-10 Subset", fontsize=14)
plt.tight_layout()
plt.show()

# 3. Preprocess (normalize + one-hot labels)
# Normalize to [0,1]
x_train = x_train.astype('float32') / 255.0
x_val   = x_val.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

num_classes = len(selected_classes)
# Map original labels (e.g., 0,1,2,3,4) to 0..num_classes-1
label_map = {c:i for i,c in enumerate(selected_classes)}
y_train_mapped = np.array([label_map[int(v)] for v in y_train])
y_val_mapped   = np.array([label_map[int(v)] for v in y_val])
y_test_mapped  = np.array([label_map[int(v)] for v in y_test])

y_train_cat = utils.to_categorical(y_train_mapped, num_classes)
y_val_cat   = utils.to_categorical(y_val_mapped, num_classes)
y_test_cat  = utils.to_categorical(y_test_mapped, num_classes)

# 4. Build a small CNN model
def build_cnn(input_shape=(32,32,3), num_classes=5):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_cnn(input_shape=x_train.shape[1:], num_classes=num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 5. Callbacks (early stopping & model checkpoint)
es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
mc = callbacks.ModelCheckpoint('best_cnn_mini.h5', monitor='val_loss', save_best_only=True)

# 6. Train
history = model.fit(
    x_train, y_train_cat,
    validation_data=(x_val, y_val_cat),
    epochs=25,
    batch_size=32,
    callbacks=[es, mc]
)

# 7. Plot training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title('Accuracy')
plt.show()

# 8. Evaluate on test set + Confusion Matrix and Classification Report
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

# Predictions
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test_mapped

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer']  # your selected classes
target_names = [f"{i} - {name}" for i, name in enumerate(class_names)]
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

# 9. Show some sample predictions (text only)
print("
Sample Predictions:")
for i in range(10):  # show first 10 samples
    print(f"Image {i+1}: True = {class_names[y_test[i]]}, Predicted = {class_names[y_pred[i]]}")
