# MNIST Digit Classification with CNN

This repository contains a Convolutional Neural Network (CNN) model built using TensorFlow and Keras for predicting different images of digits from the MNIST dataset. The model is trained to classify hand-written digits into their respective categories (0 through 9).

## Prerequisites

Before running the code, ensure that you have the required libraries installed. You can do this by running the following:

```bash
pip install numpy matplotlib tensorflow
```

## Dataset

The MNIST dataset is used for training and testing the model. It is a collection of 28x28 pixel grayscale images of handwritten digits.

## Getting Started

1. Clone this repository:

```bash
git clone https://github.com/sawanjr/mnist-cnn.git
cd mnist-cnn
```

2. Run the provided Jupyter Notebook or Python script to train and evaluate the model.

```bash
jupyter notebook mnist_cnn.ipynb
```

or

```bash
python mnist_cnn.py
```

## Model Architecture

The CNN model consists of two convolutional layers with max-pooling, followed by a flatten layer and two dense layers. The final layer uses the softmax activation function for multiclass classification.

```python
# Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## Training the Model

The model is compiled using the Adam optimizer and sparse categorical crossentropy loss. It is then trained on the MNIST training data for 5 epochs.

```python
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)
```

## Prediction and Visualization

After training, the model is used to predict digits from the test set, and the results are visualized using matplotlib.

```python
# Make predictions
predictions = model.predict(test_images)

# Visualize results for a specific image
plt.imshow(test_images[455, :, :, 0], cmap=plt.cm.binary)
plt.title(f'Predicted: {np.argmax(predictions[455])}\nTrue Label: {test_labels[455]}')
plt.show()
```

Feel free to experiment with the code and make improvements to the model for better accuracy. Enjoy experimenting with the MNIST digit classification using CNN!
