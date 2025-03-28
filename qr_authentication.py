import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For HOG feature extraction
from skimage.feature import hog
from skimage import exposure

# For traditional ML
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# For CNN using TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# For saving/loading models
import joblib

#########################################
# 1. Data Exploration and Analysis
#########################################

# Update the directories to your local paths
FIRST_PRINTS_DIR = r"C:\Users\91879\Desktop\alemeno internship\first prints"
SECOND_PRINTS_DIR = r"C:\Users\91879\Desktop\alemeno internship\second prints"

def load_images_from_folder(folder, max_images=5):
    images = []
    filenames = os.listdir(folder)
    for filename in filenames[:max_images]:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Load sample images for visualization
first_images = load_images_from_folder(FIRST_PRINTS_DIR)
second_images = load_images_from_folder(SECOND_PRINTS_DIR)

# Visualize a few images from each class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, img in enumerate(first_images):
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title('First Print')
    axes[0, i].axis('off')
for i, img in enumerate(second_images):
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].set_title('Second Print')
    axes[1, i].axis('off')
plt.tight_layout()
plt.show()

#########################################
# 2. Feature Engineering
#########################################

def extract_hog_features(image):
    image_resized = cv2.resize(image, (128, 128))
    features, hog_image = hog(
        image_resized, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=True, 
        feature_vector=True
    )
    return features, hog_image

# Example: Extract HOG features from one image and visualize
features_example, hog_image_example = extract_hog_features(first_images[0])
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(first_images[0], cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(hog_image_example, cmap='gray')
ax[1].set_title('HOG Visualization')
ax[1].axis('off')
plt.show()

#########################################
# 3. Traditional Machine Learning Pipeline
#########################################

def load_dataset(folder, label):
    features_list = []
    labels = []
    image_paths = glob.glob(os.path.join(folder, '*'))
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            feat, _ = extract_hog_features(image)
            features_list.append(feat)
            labels.append(label)
    return features_list, labels

# Load HOG features for both classes
features_first, labels_first = load_dataset(FIRST_PRINTS_DIR, 0)  # Label 0: Original
features_second, labels_second = load_dataset(SECOND_PRINTS_DIR, 1)  # Label 1: Counterfeit

# Combine features and labels
X = np.array(features_first + features_second)
y = np.array(labels_first + labels_second)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_clf = SVC(kernel='linear', probability=True)
svm_clf.fit(X_train, y_train)

# Evaluate the SVM classifier
y_pred = svm_clf.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("SVM Classification Report:\n", classification_report(y_test, y_pred))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize the confusion matrix
cm_svm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

#########################################
# 4. Deep Learning Approach (CNN)
#########################################

# The directory for CNN training data.
# Ensure this directory exists and contains subdirectories "first prints" and "second prints".
DATA_DIR = r"C:\Users\91879\Desktop\alemeno internship\data"

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"The directory '{DATA_DIR}' does not exist. "
                            "Please create it and place your 'first prints' and 'second prints' folders inside.")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Training generator
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation generator
validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Define a simple CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# Train the CNN model
history = cnn_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20
)

# Evaluate the CNN model
loss, accuracy = cnn_model.evaluate(validation_generator)
print("CNN Validation Accuracy:", accuracy)

#########################################
# 5. Save the Models
#########################################

# Save the trained SVM model using joblib
joblib.dump(svm_clf, 'svm_qr_authentication_model.joblib')
print("SVM model saved to svm_qr_authentication_model.joblib")

# Save the CNN model to an H5 file
cnn_model.save('cnn_qr_authentication_model.h5')
print("CNN model saved to cnn_qr_authentication_model.h5")

#########################################
# 6. Automatically Load and Summarize the CNN Model
#########################################

# Load the CNN model from the H5 file
loaded_cnn_model = load_model('cnn_qr_authentication_model.h5')
print("Loaded the CNN model from cnn_qr_authentication_model.h5")
loaded_cnn_model.summary()

#########################################
# 7. Automatically Load and Summarize the SVM Model
#########################################

# Load the SVM model from the JOBLIB file
loaded_svm_model = joblib.load('svm_qr_authentication_model.joblib')
print("Loaded the SVM model from svm_qr_authentication_model.joblib")
print("SVM Model Parameters:")
print(loaded_svm_model.get_params())

#########################################
# 8. Deployment Considerations (Comments)
#########################################
# For real-world deployment, consider:
# - Packaging your inference code as an API using Flask or FastAPI.
# - Optimizing model speed (e.g., using TensorFlow Lite for the CNN).
# - Implementing robust error handling and logging.
# - Ensuring security through input validation and monitoring.
# - Containerizing your solution using Docker for portability.

if __name__ == "__main__":
    print("QR Code Authentication Assignment script executed successfully.")
