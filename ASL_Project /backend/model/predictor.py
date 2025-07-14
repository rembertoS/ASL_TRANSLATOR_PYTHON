import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow as tf
from tensorflow.keras.pprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix


data_dir = 'dataset'
image_size = (64,64)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2,
    rotation_range = 15,
    zoom_range = 0.2, 
    horizontal_flip = True 
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical' , 
    subset = 'training'
)

val_generator = train_generator.flow_from_directory(
    data_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical' , 
    subset = 'validation'
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation = 'relu' , input_shape=(64, 64, 3)), 
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation ='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')

])

model.compile(

    optimizer = 'adam', 
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'],

)

history = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = 10
)

val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc:.2f}")

model.save("asl_model.h5 ")

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(['val_accuracy'], label='val accuracy' )
plt.tittle('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()





