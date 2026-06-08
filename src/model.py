import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMAGE_SIZE, CHANNELS

def build_model(num_classes: int, input_shape=None) -> models.Sequential:
    """
    Build a production-ready Convolutional Neural Network (CNN) for leaf image classification.
    
    Args:
        num_classes (int): Number of target classes for classification.
        input_shape (tuple): Shape of the input images. Defaults to config IMAGE_SIZE.
        
    Returns:
        models.Sequential: Compiled Keras model.
    """
    if input_shape is None:
        input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)
        
    model = models.Sequential([
        # Preprocessing: Rescale pixel values from [0, 255] to [0, 1]
        layers.Rescaling(1./255, input_shape=input_shape),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 3
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 4
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flattening and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5), # Regularization to prevent overfitting
        
        # Output layer with Softmax
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
