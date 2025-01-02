from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", required=True, help="Path to the training data directory")
ap.add_argument("-o", "--output", default="emotion_model.h5", help="Path to save the trained model (default: emotion_model.h5)")
args = vars(ap.parse_args())

# Build the CNN model
model = Sequential([
    # First convolutional layer: 32 filters with size 3x3, ReLU activation, input shape is 48x48 with a single channel
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    # Max pooling layer: reduces spatial dimensions, pool size is 2x2
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout layer: randomly drops 25% of neurons to reduce overfitting
    Dropout(0.25),
    # Second convolutional layer: 64 filters with size 3x3
    Conv2D(64, (3, 3), activation='relu'),
    # Max pooling layer
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout layer
    Dropout(0.25),
    # Flatten layer: converts multidimensional data into a 1D vector for the fully connected layers
    Flatten(),
    # Fully connected layer: 128 neurons, ReLU activation
    Dense(128, activation='relu'),
    # Dropout layer
    Dropout(0.5),
    # Output layer: 3 neurons (for 3 emotion categories), softmax activation to output probability distribution
    Dense(3, activation='softmax')  # 3 emotion categories
])

# Compile the model
# Using Adam optimizer with a learning rate of 0.001
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
# Using ImageDataGenerator to preprocess and augment training data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to the range [0, 1]
    rotation_range=10,  # Randomly rotate images within a range of [-10, 10] degrees
    width_shift_range=0.1,  # Randomly shift images horizontally by up to 10% of the width
    height_shift_range=0.1  # Randomly shift images vertically by up to 10% of the height
)

# Load training data from the specified directory
train_generator = train_datagen.flow_from_directory(
    args["data_dir"],  # Training data directory
    target_size=(48, 48),  # Resize images to 48x48
    batch_size=64,  # Process 64 images per batch
    color_mode="grayscale",  # Load images in grayscale
    class_mode="categorical"  # Multi-class labels, one-hot encoded
)

# Train the model
# Train for 50 epochs
model.fit(train_generator, epochs=50, steps_per_epoch=100)

# Save the trained model
# Save the model to the specified output path
model.save(args["output"])
