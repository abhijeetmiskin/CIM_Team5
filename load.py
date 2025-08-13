import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2 # For transfer learning example
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
# IMPORTANT:
# Replace "traffic_sign_dataset_split" with the actual path
# where your 'train' and 'test' folders are located.
dataset_split_root = "/home/user/Desktop"
image_size = (150, 150) # All images will be resized to this (width, height)
batch_size = 32
epochs = 15 # Number of training epochs (adjust as needed)
learning_rate = 0.001 # Initial learning rate for the optimizer
use_transfer_learning = True # Set to False to use a simple custom CNN instead of MobileNetV2

# Define paths to train and test directories based on the root
train_dir = os.path.join(dataset_split_root, 'train')
test_dir = os.path.join(dataset_split_root, 'test')

# --- 1. Data Loading and Augmentation ---
print("--- Setting up Data Generators ---")

# Data Augmentation and Preprocessing for Training
# Rescale pixel values to [0, 1] for neural network input
# Apply various random transformations to augment the training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only Rescaling for Test Data (no augmentation for consistent evaluation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories using flow_from_directory
# It infers class labels from subfolder names.
# class_mode='binary' since you have two classes ('normal_signs', 'damaged_signs').
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary', # For 2 classes, outputs 0 or 1
    seed=42 # Set a seed for reproducibility of shuffles and transformations
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False, # IMPORTANT: Do NOT shuffle test data for consistent evaluation metrics
    seed=42
)

# Print the mapping of class names to numerical indices
# This helps you understand what 0 and 1 represent in your predictions
print(f"Class indices: {train_generator.class_indices}")
# Expected output might be something like: {'damaged_signs': 0, 'normal_signs': 1}
# or vice-versa, depending on alphabetical order.

# --- 2. Model Definition ---
print("\n--- Building the Model ---")

if use_transfer_learning:
    print("Using Transfer Learning (MobileNetV2 pre-trained on ImageNet)...")
    # Load MobileNetV2 base model without its top (classification) layer.
    # weights='imagenet' loads the pre-trained weights.
    # include_top=False means we don't include the final classification layers.
    # input_shape must match the image_size and 3 color channels.
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

    # Freeze the layers of the base model so they are not updated during initial training.
    # This prevents destroying the learned features from ImageNet.
    base_model.trainable = False

    # Add your custom classification head on top of the base model's output.
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Reduces spatial dimensions, suitable for connecting to dense layers.
    x = Dense(128, activation='relu')(x) # A dense hidden layer
    predictions = Dense(1, activation='sigmoid')(x) # Output layer for binary classification

    # Create the full model by connecting the base model's input to our new output.
    model = Model(inputs=base_model.input, outputs=predictions)

else:
    print("Using a Custom Simple CNN from scratch...")
    # Define a sequential CNN model
    model = models.Sequential([
        # Convolutional layer with 32 filters, 3x3 kernel, ReLU activation
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
        layers.MaxPooling2D((2, 2)), # Max pooling to reduce dimensionality
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(), # Flatten 2D feature maps into a 1D vector
        layers.Dense(128, activation='relu'), # A dense hidden layer
        layers.Dense(1, activation='sigmoid') # Output layer: 1 unit with sigmoid for binary classification
    ])

# Print a summary of the model's architecture
model.summary()

# --- 3. Model Compilation ---
print("\n--- Compiling the Model ---")
# Use Adam optimizer with a specified learning rate
# binary_crossentropy is the standard loss for binary classification
# 'accuracy' is the metric to monitor during training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- 4. Model Training ---
print("\n--- Training the Model ---")

# Define callbacks for better training control and saving the best model
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',         # Monitor validation loss
        patience=5,                 # Stop if val_loss doesn't improve for 5 consecutive epochs
        restore_best_weights=True   # Restore model weights from the best epoch
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_traffic_sign_model.h5', # Path to save the best model file
        monitor='val_accuracy',     # Monitor validation accuracy
        save_best_only=True,        # Only save the model if it's the best so far
        mode='max',                 # We want to maximize validation accuracy
        verbose=1
    )
    # You can add more callbacks here, e.g., ReduceLROnPlateau
]

# Train the model using the data generators
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size, # Number of steps per epoch (total samples / batch size)
    epochs=epochs, # Number of epochs to train for
    validation_data=test_generator, # Data to evaluate on at the end of each epoch
    validation_steps=test_generator.samples // batch_size, # Number of validation steps
    callbacks=callbacks # Apply the defined callbacks
)

# Load the best model saved by ModelCheckpoint for final evaluation
# This ensures you're evaluating the model that performed best on the validation set
try:
    model = tf.keras.models.load_model('best_traffic_sign_model.h5')
    print("\nLoaded best model for final evaluation.")
except Exception as e:
    print(f"\nCould not load 'best_traffic_sign_model.h5': {e}. Using the model from the last epoch.")


# --- 5. Model Evaluation ---
print("\n--- Evaluating the Model ---")

# Evaluate the model's performance on the test set
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get predictions for detailed metrics like classification report and confusion matrix
print("\n--- Generating Predictions for detailed metrics ---")
test_generator.reset() # Reset the test generator to ensure predictions are in the correct order
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

# Convert probabilities (sigmoid output) to binary class labels (0 or 1)
# A probability >= 0.5 is rounded to 1, otherwise to 0
y_pred_binary = np.round(predictions).flatten()

# Get the true labels from the test generator
y_true = test_generator.classes

# Get the class names in the correct order as per the generator's indices
idx_to_class = {v: k for k, v in test_generator.class_indices.items()}
# Ensure target_names align with 0 and 1, assuming damaged_signs is 0 and normal_signs is 1
# Adjust if your generator's class_indices are different
target_names = [idx_to_class[0], idx_to_class[1]]

print("\n--- Classification Report ---")
# Provides precision, recall, f1-score for each class
print(classification_report(y_true, y_pred_binary, target_names=target_names))

print("\n--- Confusion Matrix ---")
# Shows true positives, true negatives, false positives, false negatives
cm = confusion_matrix(y_true, y_pred_binary)
print(cm)
print(f"True Negatives ({target_names[0]} correctly predicted as {target_names[0]}): {cm[0, 0]}")
print(f"False Positives (Actual {target_names[0]} predicted as {target_names[1]}): {cm[0, 1]}")
print(f"False Negatives (Actual {target_names[1]} predicted as {target_names[0]}): {cm[1, 0]}")
print(f"True Positives ({target_names[1]} correctly predicted as {target_names[1]}): {cm[1, 1]}")


# --- Plotting Training History ---
print("\n--- Plotting Training History ---")
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout() # Adjusts plot parameters for a tight layout
plt.show()

print("\nTraining and evaluation complete.")
print(f"Best model saved as: best_traffic_sign_model.h5")