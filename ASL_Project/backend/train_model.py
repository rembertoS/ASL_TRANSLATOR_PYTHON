#!/usr/bin/env python3
"""
Training script for ASL Recognition Model
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def create_model(num_classes):
    """Create the CNN model architecture"""
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(data_dir='../asl_alphabet_train', epochs=20, batch_size=64):
    """Train the ASL recognition model"""
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found!")
        print("Please create the dataset structure:")
        print("asl_alphabet_train/")
        print("├── A/")
        print("├── B/")
        print("├── C/")
        print("└── ... (one folder per letter)")
        return None
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=False,  # Don't flip ASL signs
        width_shift_range=0.05,
        height_shift_range=0.05
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    print(f" Found {train_generator.num_classes} classes")
    print(f" Training samples: {train_generator.samples}")
    print(f" Validation samples: {val_generator.samples}")
    
    # Create and train model
    model = create_model(train_generator.num_classes)
    
    # Callbacks for better training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'model/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    print(" Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    val_loss, val_acc = model.evaluate(val_generator)
    print(f" Final validation accuracy: {val_acc:.2f}")
    
    # Save model
    model_path = "model/asl_model.h5"
    model.save(model_path)
    print(f" Model saved to {model_path}")
    
    # Plot training history
    plot_training_history(history)
    
    return model

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    print("ASL Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists('../asl_alphabet_train'):
        print("No dataset found!")
        print("\n To create a dataset:")
        print("1. Create an 'asl_alphabet_train' folder")
        print("2. Create subfolders for each letter (A, B, C, ..., Z)")
        print("3. Add ASL images to each folder")
        print("4. Run this script again")
        return
    
    # Train the model
    model = train_model()
    
    if model:
        print("\n Training completed successfully!")
        print("You can now run the backend with: python main.py")

if __name__ == "__main__":
    main() 