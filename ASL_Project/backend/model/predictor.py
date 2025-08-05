import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class ASLPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            
            model_path = os.path.join(os.path.dirname(__file__), 'asl_model.h5')
        self.model_path = model_path
        self.model = None
        self.image_size = (64, 64)
        self.class_names = [
            'A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained ASL model"""
        try:
            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 0:
                self.model = load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Model file {self.model_path} not found or empty. Using dummy model.")
                self.model = self.create_dummy_model()
        except Exception as e:
            print(f"Error loading model: {e}. Using dummy model.")
            self.model = self.create_dummy_model()
    
    def create_dummy_model(self):
        """Create a dummy model for testing when no trained model is available"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(29, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        try:
            # Load and resize image
            img = load_img(image_path, target_size=self.image_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize
            return img_array
        except Exception as e:
            raise Exception(f"Error preprocessing image: {e}")
    
    def predict(self, image_path):
        """Predict ASL sign from image"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Get the predicted letter
            predicted_letter = self.class_names[predicted_class]
            
            return f"{predicted_letter} (confidence: {confidence:.2f})"
            
        except Exception as e:
            raise Exception(f"Prediction failed: {e}")


def train_model():
    data_dir = '../asl_alphabet_train'
    image_size = (64, 64)
    batch_size = 32

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10
    )

    model.save(os.path.join(os.path.dirname(__file__), 'asl_model.h5'))
    return model

if __name__ == '__main__':
    train_model()





