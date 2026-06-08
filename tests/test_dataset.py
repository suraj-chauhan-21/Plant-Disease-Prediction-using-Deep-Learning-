import pytest
from src.config import CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, VALIDATION_SPLIT

def test_config_dimensions():
    """
    Validates dataset configuration integrity.
    """
    assert len(CLASS_NAMES) == 38, "The system expects exactly 38 categories from PlantVillage dataset"
    assert isinstance(IMAGE_SIZE, tuple) and len(IMAGE_SIZE) == 2, "IMAGE_SIZE must be a 2-tuple (width, height)"
    assert IMAGE_SIZE[0] > 0 and IMAGE_SIZE[1] > 0, "Dimensions must be positive integers"
    assert BATCH_SIZE > 0, "Batch size must be a positive integer"
    assert 0.0 < VALIDATION_SPLIT < 1.0, "Validation split must be a decimal fraction between 0 and 1"
