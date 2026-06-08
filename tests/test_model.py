import pytest
import numpy as np
import tensorflow as tf
from src.model import build_model
from src.config import CLASS_NAMES

def test_model_output_shape():
    """
    Test that the CNN model compiles and returns the correct output shape
    matching the configured number of classes.
    """
    num_classes = len(CLASS_NAMES)
    model = build_model(num_classes=num_classes)
    
    # Check that model compile succeeded
    assert model is not None
    assert len(model.layers) > 0
    
    # Create mock batch of 2 RGB images of size 128x128
    mock_input = np.random.rand(2, 128, 128, 3).astype(np.float32) * 255.0
    predictions = model.predict(mock_input)
    
    # Shape should be (batch_size, num_classes)
    assert predictions.shape == (2, num_classes)
    
    # Softmax output: probabilities should sum to ~1 for each sample
    for i in range(len(predictions)):
        assert np.isclose(np.sum(predictions[i]), 1.0, atol=1e-5)
