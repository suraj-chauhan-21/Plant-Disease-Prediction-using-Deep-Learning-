import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from src.config import CLASS_NAMES, MODEL_PATH, IMAGE_SIZE

def preprocess_image(image_path: str, target_size=IMAGE_SIZE):
    """
    Load and preprocess a single image for prediction.
    
    Args:
        image_path (str): Filepath of the image to preprocess.
        target_size (tuple): Target dimensions (width, height) expected by the model.
        
    Returns:
        np.ndarray: Preprocessed image batch array with dimensions (1, height, width, channels).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
        
    img = Image.open(image_path)
    img = img.convert("RGB")
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 128, 128, 3)
    return img_batch

def predict_disease(image_path: str, model_path=MODEL_PATH, threshold=0.0):
    """
    Load model and run prediction on a single image.
    
    Args:
        image_path (str): Path to the image file.
        model_path (str): Path to the saved Keras model file.
        threshold (float): Confidence threshold (0.0 to 1.0) under which to reject.
        
    Returns:
        dict: Prediction results including crop, disease status, and confidence.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model weights file not found at: {model_path}. "
            "Please run the training pipeline first."
        )
        
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess image
    processed_img = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(processed_img)
    confidence_scores = predictions[0]
    predicted_idx = np.argmax(confidence_scores)
    confidence = float(confidence_scores[predicted_idx])
    
    raw_class_name = CLASS_NAMES[predicted_idx]
    
    # Parsed labels
    parts = raw_class_name.split("___")
    crop = parts[0].replace("_", " ").capitalize()
    disease_status = parts[1].replace("_", " ").capitalize()
    
    if confidence < threshold:
        return {
            "status": "Uncertain",
            "crop": crop,
            "disease_status": "Detection confidence is below threshold.",
            "confidence": confidence,
            "raw_class": raw_class_name
        }
        
    return {
        "status": "Success",
        "crop": crop,
        "disease_status": disease_status,
        "confidence": confidence,
        "raw_class": raw_class_name
    }

def main():
    parser = argparse.ArgumentParser(description="Predict plant disease from leaf image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the leaf image.")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to trained .keras model.")
    parser.add_argument("--threshold", type=float, default=0.4, help="Min confidence threshold.")
    args = parser.parse_args()
    
    try:
        result = predict_disease(args.image, model_path=args.model, threshold=args.threshold)
        print("\n" + "=" * 40)
        print("         DIAGNOSIS RESULTS")
        print("=" * 40)
        print(f"File:        {os.path.basename(args.image)}")
        print(f"Crop Type:   {result['crop']}")
        print(f"Condition:   {result['disease_status']}")
        print(f"Confidence:  {result['confidence'] * 100:.2f}%")
        print("=" * 40 + "\n")
    except Exception as e:
        print(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
