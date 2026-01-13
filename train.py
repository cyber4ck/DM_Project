import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import json
import os

# Configuration
DATASET_DIR = 'PlantVillage'  # Folder containing class folders
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Augmentation & Loading
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save Class Names to JSON (Crucial for mapping predictions)
class_indices = train_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}
with open('classes.json', 'w') as f:
    json.dump(class_names, f)
print("Class labels saved to classes.json")

# Build Simple CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_indices), activation='softmax') # Output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
print("Starting training...")
model.fit(train_generator, validation_data=validation_generator, epochs=5)

# Save Model
model.save('model.h5')
print("Model saved as model.h5")