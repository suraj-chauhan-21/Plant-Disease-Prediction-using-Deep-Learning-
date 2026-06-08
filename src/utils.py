import os
import json
import matplotlib.pyplot as plt

def plot_training_history(history, save_path='training_history.png'):
    """
    Plots training and validation accuracy/loss metrics and saves the plot as an image.
    
    Args:
        history: Keras History object (or dictionary of history history).
        save_path (str): File path to save the generated plot.
    """
    # Extract metrics
    if hasattr(history, 'history'):
        metrics = history.history
    else:
        metrics = history

    acc = metrics.get('accuracy', [])
    val_acc = metrics.get('val_accuracy', [])
    loss = metrics.get('loss', [])
    val_loss = metrics.get('val_loss', [])
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#2e7d32', linewidth=2)
    if val_acc:
        plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#ff9800', linewidth=2)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#c62828', linewidth=2)
    if val_loss:
        plt.plot(epochs_range, val_loss, label='Validation Loss', color='#0277bd', linewidth=2)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Metrics plot successfully saved to '{save_path}'")

def save_metrics_to_json(history, save_path='metrics.json'):
    """
    Saves history metrics dictionary to a JSON file.
    
    Args:
        history: Keras History object or metrics dictionary.
        save_path (str): File path to save JSON metrics.
    """
    if hasattr(history, 'history'):
        metrics = history.history
    else:
        metrics = history
        
    # Cast float values to serializable types
    serializable_metrics = {}
    for key, val in metrics.items():
        serializable_metrics[key] = [float(x) for x in val]
        
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=4)
    print(f"Metrics history saved to '{save_path}'")
