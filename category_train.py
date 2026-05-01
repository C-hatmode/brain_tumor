import os
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURATION
# -------------------------------
DATASET_DIR = "C:/Users/sakshi/Documents/brain_tumor_detection/datasets/tumor_category_nifti"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = "C:/Users/sakshi/Documents/brain_tumor_detection/models/category_model.h5"

# -------------------------------
# LOAD NIfTI DATA
# -------------------------------
def load_nifti_data(dataset_dir):
    X = []
    y = []
    labels = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
    label_dict = {name: idx for idx, name in enumerate(labels)}
    print("Found categories:", labels)

    for label in labels:
        path = os.path.join(dataset_dir, label)
        for file in tqdm(os.listdir(path), desc=f"Loading {label}"):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                file_path = os.path.join(path, file)
                try:
                    nifti_img = nib.load(file_path)
                    img_data = nifti_img.get_fdata()

                    # Take middle slice from 3D MRI volume
                    if img_data.ndim == 3:
                        mid_slice = img_data.shape[2] // 2
                        img_2d = img_data[:, :, mid_slice]
                    else:
                        img_2d = img_data

                    # Resize and expand dims for CNN
                    img_2d = cv2.resize(img_2d, (IMG_SIZE, IMG_SIZE))
                    img_2d = np.expand_dims(img_2d, axis=-1)

                    X.append(img_2d)
                    y.append(label_dict[label])
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    X = np.array(X) / 255.0  # normalize
    y = np.array(y)
    return X, y, labels

# -------------------------------
# LOAD DATA
# -------------------------------
print("Loading dataset...")
X, y, labels = load_nifti_data(DATASET_DIR)
print(f"Dataset Loaded: {X.shape[0]} samples")

# -------------------------------
# SPLIT DATA
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# -------------------------------
# BUILD CNN MODEL
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    BatchNormalization(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# TRAIN MODEL
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# -------------------------------
# EVALUATE & SAVE
# -------------------------------
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.4f}")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}")

# -------------------------------
# PLOT TRAINING PERFORMANCE
# -------------------------------
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Model Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title('Model Loss')
plt.legend()
plt.show()
