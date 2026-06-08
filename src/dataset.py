import os
import tensorflow as tf
from src.config import DATASET_PATH, IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT, SEED

def get_data_augmentation() -> tf.keras.Sequential:
    """
    Returns a data augmentation pipeline to prevent overfitting during training.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ])

def load_dataset(dataset_path=None, batch_size=None, image_size=None):
    """
    Loads training and validation datasets from a directory structure.
    
    Args:
        dataset_path (str): Path to root training directory.
        batch_size (int): Size of batches.
        image_size (tuple): Width and height of target images.
        
    Returns:
        tuple: (train_dataset, validation_dataset, class_names)
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH
    if batch_size is None:
        batch_size = BATCH_SIZE
    if image_size is None:
        image_size = IMAGE_SIZE
        
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_path}' not found. "
            "Please configure the dataset folder correctly."
        )
        
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=image_size,
        batch_size=batch_size
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=image_size,
        batch_size=batch_size
    )
    
    class_names = train_ds.class_names
    
    # Apply performance prefetching and caching
    AUTOTUNE = tf.data.AUTOTUNE
    
    # We apply augmentations using Keras layers either in model definition or pipeline.
    # In TensorFlow, it is typically cleaner to include augmentation layers directly 
    # in the model definition so that it is automatically exported with the model 
    # and bypassed during evaluation.
    
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names
