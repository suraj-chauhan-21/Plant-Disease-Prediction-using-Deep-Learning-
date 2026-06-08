import argparse
import os
import tensorflow as tf
from src.config import EPOCHS, BATCH_SIZE, MODEL_PATH, DATASET_PATH
from src.dataset import load_dataset
from src.model import build_model
from src.utils import plot_training_history, save_metrics_to_json

def train_pipeline(dataset_path: str, epochs: int, batch_size: int, model_output_path: str):
    """
    Executes the training pipeline from data loading to model evaluation and saving.
    
    Args:
        dataset_path (str): Path to raw images.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        model_output_path (str): Path to save the final trained model.
    """
    print("=" * 60)
    print("Starting Plant Disease Detection System Training Pipeline")
    print("=" * 60)
    
    # 1. Load Data
    print(f"Loading dataset from: {dataset_path}...")
    try:
        train_ds, val_ds, class_names = load_dataset(
            dataset_path=dataset_path,
            batch_size=batch_size
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your dataset path and directory structure.")
        return

    # 2. Build Model
    print(f"Building Convolutional Neural Network for {len(class_names)} classes...")
    model = build_model(num_classes=len(class_names))
    model.summary()

    # 3. Callbacks (Early Stopping and Model Checkpointing)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    # Ensure parent directory of model output exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_output_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # 4. Train Model
    print(f"Beginning training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint]
    )

    # 5. Save Model Checkpoint
    # Even if checkpoint saved the best weights, we do a final save
    print(f"Saving final trained model to: {model_output_path}...")
    model.save(model_output_path)
    
    # 6. Save Plot and Metrics
    metrics_plot_path = os.path.join(os.path.dirname(model_output_path), "training_history.png")
    metrics_json_path = os.path.join(os.path.dirname(model_output_path), "metrics.json")
    
    plot_training_history(history, save_path=metrics_plot_path)
    save_metrics_to_json(history, save_path=metrics_json_path)
    
    print("=" * 60)
    print("Training Pipeline Executed Successfully!")
    print(f"Saved model: {model_output_path}")
    print(f"Saved training history plot: {metrics_plot_path}")
    print(f"Saved JSON metrics log: {metrics_json_path}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection CNN Model.")
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH, help="Path to training dataset.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Input batch size for training.")
    parser.add_argument("--output-model", type=str, default=MODEL_PATH, help="Output destination for model.")
    args = parser.parse_args()
    
    train_pipeline(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_output_path=args.output_model
    )

if __name__ == "__main__":
    main()
