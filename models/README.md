# Model Directory

This directory stores trained deep learning model weights and history checkpoint files. Large binary weight files are ignored by Git (using `.gitignore`) to keep the repository lightweight.

## Saved Assets

When running `python src/train.py`, the following assets are generated here:

1. **`plant_disease_model.keras`**: The saved trained Keras model, containing both weights and architecture.
2. **`training_history.png`**: Visualization showing loss and accuracy plots for both training and validation loops.
3. **`metrics.json`**: A log of numerical training and validation metrics for each epoch (useful for comparison and tracking).

## Model Details

- **Input Dimension:** `128 x 128 x 3` (RGB Leaf Specimen Images)
- **Architecture:** 4-Block CNN with Batch Normalization and Dropout layers for regularization.
- **Classes:** 38 distinct crop disease/healthy labels.
- **Framework:** TensorFlow / Keras
