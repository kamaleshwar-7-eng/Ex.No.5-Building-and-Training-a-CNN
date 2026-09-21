# Ex.No.5 – Building and Training a CNN

## AIM

To build and train a Convolutional Neural Network (CNN) using TensorFlow/Keras for recognizing sign language digits from 0 to 9 and evaluate its performance using accuracy, classification report, and confusion matrix.

---

## PROCEDURE

### Step 1: Import Libraries

- Import TensorFlow and Keras for building and training the CNN.
- Import NumPy for numerical operations.
- Import Matplotlib and Seaborn for visualization.
- Import Scikit-learn metrics for model evaluation.
- Import Google Colab utilities for accessing Google Drive and uploading test images.

### Step 2: Mount Google Drive

- Mount Google Drive in Google Colab.
- Locate the uploaded Sign Language Digits dataset ZIP file.

### Step 3: Extract the Dataset

- Extract the dataset ZIP file into the Colab environment.
- Locate the `Dataset` folder.
- The dataset contains separate folders for digits 0 to 9.

### Step 4: Create Training and Validation Datasets

- Load the images using `image_dataset_from_directory()`.
- Resize all images to 64 × 64 pixels.
- Divide the dataset into 80% training data and 20% validation data.
- Use a batch size of 32.

### Step 5: Build the CNN Model

- Create a Sequential CNN architecture.
- Normalize image pixel values from 0–255 to 0–1.
- Apply three convolutional layers to extract image features.
- Apply max-pooling layers to reduce spatial dimensions.
- Flatten the extracted features.
- Use a dense layer for classification.
- Use a 10-unit Softmax output layer for digits 0–9.

### Step 6: Compile the CNN

- Use the Adam optimizer.
- Use Sparse Categorical Cross-Entropy as the loss function.
- Use accuracy as the evaluation metric.

### Step 7: Train the Model

- Train the CNN using the training dataset.
- Validate the model using the validation dataset.
- Train the model for 15 epochs.

### Step 8: Evaluate the Model

- Generate predictions for the validation images.
- Calculate the overall classification accuracy.
- Generate a classification report containing precision, recall, and F1-score.
- Generate a confusion matrix for the ten digit classes.

### Step 9: Visualize Training Performance

- Plot training and validation accuracy.
- Plot training and validation loss.
- Analyze the training performance of the CNN.

### Step 10: Test with a New Image

- Upload a new sign language digit image.
- Resize the image to 64 × 64 pixels.
- Pass the image through the trained CNN.
- Display the predicted digit and confidence percentage.

---

## THEORY

### 1. Convolutional Neural Network

A Convolutional Neural Network (CNN) is a deep learning model mainly used for image classification and computer vision tasks. It automatically learns important features such as edges, shapes, textures, and patterns from images.

### 2. Image Rescaling

Image pixels normally have values between 0 and 255. Rescaling converts these values into the range 0 to 1.

**Formula:**

`Normalized Pixel = Pixel Value / 255`

This helps the neural network process the image efficiently.

### 3. Convolution Layer

The convolution layer uses filters to extract important features from an image.

In this experiment:

- The first convolution layer extracts basic edges and shapes.
- The second convolution layer extracts more complex hand patterns.
- The third convolution layer extracts higher-level structural features.

### 4. ReLU Activation Function

ReLU stands for Rectified Linear Unit. It introduces non-linearity into the neural network.

**Formula:**

`ReLU(x) = max(0, x)`

### 5. Max Pooling

Max pooling reduces the spatial dimensions of feature maps while retaining important features.

A 2 × 2 pooling window is used in this experiment.

### 6. Flatten Layer

The Flatten layer converts the two-dimensional feature maps into a one-dimensional vector before passing them to the dense layer.

### 7. Dense Layer

The dense layer learns relationships between the extracted features and the different digit classes.

The CNN uses a dense layer containing 64 neurons with ReLU activation.

### 8. Softmax Output Layer

The final layer contains 10 neurons representing the digit classes:

`0, 1, 2, 3, 4, 5, 6, 7, 8, 9`

Softmax produces a probability for each class. The class with the highest probability is selected as the predicted digit.

### 9. Adam Optimizer

Adam (Adaptive Moment Estimation) is an optimization algorithm used to update the weights of the neural network during training.

### 10. Sparse Categorical Cross-Entropy

Sparse categorical cross-entropy measures the difference between the actual digit label and the predicted class probabilities. It is suitable when class labels are represented as integers.

### 11. Accuracy

Accuracy represents the percentage of correctly classified images.

**Formula:**

`Accuracy = (Correct Predictions / Total Predictions) × 100`

### 12. Confusion Matrix

A confusion matrix compares the actual classes with the predicted classes. It helps identify which digit classes are correctly or incorrectly classified.

### 13. Classification Report

The classification report provides:

- **Precision** – correctness of predicted classes.
- **Recall** – ability to identify samples of a class.
- **F1-score** – combined measure of precision and recall.
- **Support** – number of samples in each class.

---

## PROGRAM

```python
# ============================================================
# CNN - SIGN LANGUAGE DIGITS CLASSIFICATION
# ============================================================

import os
import glob
import zipfile
import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import drive, files


# ============================================================
# 1. MOUNT GOOGLE DRIVE
# ============================================================

drive.mount('/content/drive')


# ============================================================
# 2. FIND ZIP FILE AUTOMATICALLY IN GOOGLE DRIVE
# ============================================================

zip_files = glob.glob(
    '/content/drive/MyDrive/**/*.zip',
    recursive=True
)

print("ZIP files found in Google Drive:")

if len(zip_files) == 0:
    raise FileNotFoundError(
        "No ZIP file was found in Google Drive."
    )

for i, path in enumerate(zip_files):
    print(f"{i}: {path}")


# ============================================================
# 3. SELECT THE SIGN LANGUAGE DATASET ZIP
# ============================================================

dataset_zips = [
    path for path in zip_files
    if 'sign' in os.path.basename(path).lower()
    or 'digit' in os.path.basename(path).lower()
]

if len(dataset_zips) > 0:
    zip_path = dataset_zips[0]
else:
    zip_path = zip_files[0]

print("\nUsing ZIP file:")
print(zip_path)


# ============================================================
# 4. EXTRACT DATASET
# ============================================================

extract_path = '/content/cnn_dataset'

# Remove previous extraction
if os.path.exists(extract_path):
    shutil.rmtree(extract_path)

os.makedirs(extract_path, exist_ok=True)

print("\nExtracting dataset...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction completed.")


# ============================================================
# 5. FIND THE DATASET FOLDER AUTOMATICALLY
# ============================================================

dataset_candidates = []

for root, dirs, files_list in os.walk(extract_path):

    if os.path.basename(root).lower() == 'dataset':

        # Check whether this folder contains digit folders
        digit_folders = [
            str(i) for i in range(10)
            if str(i) in dirs
        ]

        if len(digit_folders) >= 2:
            dataset_candidates.append(root)


if len(dataset_candidates) == 0:

    print("\nCould not automatically find Dataset folder.")
    print("\nExtracted folder structure:")

    for root, dirs, files_list in os.walk(extract_path):
        level = root.replace(extract_path, '').count(os.sep)

        if level <= 3:
            print(root)

    raise FileNotFoundError(
        "Dataset folder containing digit classes 0-9 was not found."
    )


data_dir = dataset_candidates[0]

print("\nDataset path:")
print(data_dir)


# ============================================================
# 6. CHECK DIGIT FOLDERS
# ============================================================

print("\nDigit folders:")

for i in range(10):

    folder = os.path.join(
        data_dir,
        str(i)
    )

    if os.path.exists(folder):

        image_count = len(
            [
                f for f in os.listdir(folder)
                if f.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.bmp')
                )
            ]
        )

        print(
            f"Digit {i}: {image_count} images"
        )

    else:

        print(
            f"WARNING: Digit {i} folder not found"
        )


# ============================================================
# 7. CREATE TRAINING DATASET
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(

    data_dir,

    validation_split=0.2,

    subset='training',

    seed=123,

    image_size=(64, 64),

    batch_size=32
)


# ============================================================
# 8. CREATE VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(

    data_dir,

    validation_split=0.2,

    subset='validation',

    seed=123,

    image_size=(64, 64),

    batch_size=32
)


# ============================================================
# 9. DISPLAY CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\nClass names:")
print(class_names)


# ============================================================
# 10. IMPROVE DATA PIPELINE PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 11. BUILD CNN MODEL
# ============================================================

model = models.Sequential([

    # Input
    layers.Input(
        shape=(64, 64, 3)
    ),

    # Normalize pixels from 0-255 to 0-1
    layers.Rescaling(
        1.0 / 255
    ),

    # CNN Layer 1
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # CNN Layer 2
    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # CNN Layer 3
    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Flatten
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(
        64,
        activation='relu'
    ),

    # Output Layer
    layers.Dense(
        10,
        activation='softmax'
    )
])


# ============================================================
# 12. DISPLAY MODEL SUMMARY
# ============================================================

print("\nCNN Model Summary:")

model.summary()


# ============================================================
# 13. COMPILE MODEL
# ============================================================

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print("\nStarting CNN training...")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=15
)


# ============================================================
# 15. PREDICT VALIDATION DATA
# ============================================================

print("\nGenerating predictions...")

y_true = []

y_pred = []

for images, labels in val_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        np.argmax(
            predictions,
            axis=1
        )
    )


y_true = np.array(y_true)

y_pred = np.array(y_pred)


# ============================================================
# 16. CALCULATE ACCURACY
# ============================================================

accuracy = (
    np.mean(
        y_true == y_pred
    ) * 100
)

print(
    f"\nValidation Accuracy: "
    f"{accuracy:.2f}%"
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=class_names,

    yticklabels=class_names
)

plt.title(
    'CNN Confusion Matrix'
)

plt.ylabel(
    'Actual'
)

plt.xlabel(
    'Predicted'
)

plt.show()


# ============================================================
# 19. TRAINING VS VALIDATION ACCURACY
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title(
    'Training and Validation Accuracy'
)

plt.xlabel(
    'Epoch'
)

plt.ylabel(
    'Accuracy'
)

plt.legend()

plt.show()


# ============================================================
# 20. TRAINING VS VALIDATION LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title(
    'Training and Validation Loss'
)

plt.xlabel(
    'Epoch'
)

plt.ylabel(
    'Loss'
)

plt.legend()

plt.show()


# ============================================================
# 21. UPLOAD NEW IMAGE FOR PREDICTION
# ============================================================

print("\nUpload an image of a sign-language digit:")

uploaded = files.upload()


# ============================================================
# 22. PREDICT UPLOADED IMAGE
# ============================================================

for filename in uploaded.keys():

    # Load image
    img = tf.keras.utils.load_img(
        filename,
        target_size=(64, 64)
    )

    # Convert image to array
    img_array = tf.keras.utils.img_to_array(
        img
    )

    # Add batch dimension
    img_array = tf.expand_dims(
        img_array,
        0
    )

    # Predict
    predictions = model.predict(
        img_array,
        verbose=0
    )

    # Predicted class
    predicted_class = np.argmax(
        predictions[0]
    )

    # Confidence
    confidence = (
        np.max(
            predictions[0]
        ) * 100
    )

    # Display result
    plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(img)

    plt.title(
        f"Prediction: {predicted_class} "
        f"({confidence:.2f}% Confidence)"
    )

    plt.axis('off')

    plt.show()
```

## OUTPUT
<img width="1160" height="776" alt="image" src="https://github.com/user-attachments/assets/c225abd1-72b0-4753-9d94-2525017ab062" />
<img width="997" height="883" alt="image" src="https://github.com/user-attachments/assets/7dbbee53-1b72-4853-bf5b-add323087b9b" />
<img width="864" height="565" alt="image" src="https://github.com/user-attachments/assets/44e6dd6a-e0e5-4f4b-bc25-aa039f8734d8" />
<img width="694" height="625" alt="image" src="https://github.com/user-attachments/assets/1acab55a-1b48-49c7-bc46-10b1974f816a" />


## CONCLUSION

Thus, a Convolutional Neural Network (CNN) was successfully built and trained using TensorFlow/Keras for sign language digit classification. The CNN learned relevant image features through convolution and pooling layers and classified images into ten digit classes from 0 to 9. The model was evaluated using accuracy, classification report, and confusion matrix, and was also used to predict a new sign language digit image.
