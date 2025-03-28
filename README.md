__________________________________________________________________________________________________________
The dataset link: https://drive.google.com/drive/folders/1YjCVaQy8GQi5XWXf88PtgW8qOaplS_Bd?usp=sharing
___________________________________________________________________________________________________________
In this project I've implemented CNN using SVM.
1. Feature Extraction with CNN
A CNN model is built using Keras, consisting of convolutional layers, max-pooling layers, and fully connected layers.

*The convolutional layers detect spatial features in the QR code images, such as patterns and unique structures.

*The Flatten layer converts the extracted features into a format suitable for classification.

2. CNN Training for QR Code Recognition
*The CNN model is trained on QR code images, learning to distinguish between different classes (e.g., valid vs. invalid QR codes).

*Backpropagation and gradient descent are used to update the network’s weights to minimize classification errors.

3. Using CNN for Feature Extraction in SVM
*Instead of directly using the CNN for classification, the feature representations from the Flatten layer are extracted.

*These features are then used as input to a Support Vector Machine (SVM) classifier.

The SVM is trained on these high-dimensional CNN-generated features rather than raw images, making classification more efficient.

