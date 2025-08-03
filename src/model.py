"""
Rock Paper Scissors CNN Model Architecture and Loading Module

This module contains the CNN model architecture for Rock Paper Scissors classification
and utilities for loading trained models. This is part of the ML summative assignment.

Author: ML Student
Date: August 2025
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
NUM_CLASSES = 3
CLASS_NAMES = ['paper', 'rock', 'scissors']

def create_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    """
    Create a CNN model optimized for rock-paper-scissors classification.
    
    This model uses modern deep learning techniques including:
    - Batch Normalization for training stability
    - Dropout for regularization
    - L2 Regularization to prevent overfitting
    - GlobalAveragePooling2D to reduce parameters
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Compiled CNN model ready for training or inference
    """
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape, name='input_layer'),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(0.001),
                     name='conv2d_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.MaxPooling2D(2, 2, name='maxpool_1'),
        layers.Dropout(0.25, name='dropout_1'),
        
        # Second Convolutional Block  
        layers.Conv2D(64, (3, 3), activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001),
                     name='conv2d_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.MaxPooling2D(2, 2, name='maxpool_2'),
        layers.Dropout(0.25, name='dropout_2'),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001),
                     name='conv2d_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.MaxPooling2D(2, 2, name='maxpool_3'),
        layers.Dropout(0.25, name='dropout_3'),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001),
                     name='conv2d_4'),
        layers.BatchNormalization(name='batch_norm_4'),
        layers.GlobalAveragePooling2D(name='global_avg_pool'),  # Better than Flatten
        
        # Dense layers
        layers.Dense(512, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    name='dense_1'),
        layers.BatchNormalization(name='batch_norm_5'),
        layers.Dropout(0.5, name='dropout_4'),
        
        layers.Dense(128, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    name='dense_2'),
        layers.Dropout(0.3, name='dropout_5'),
        
        # Output layer (3 classes: rock, paper, scissors)
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ], name='RockPaperScissors_CNN')
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    logger.info(f"Created CNN model with {model.count_params():,} parameters")
    return model


def load_trained_model(model_path, model_format='keras'):
    """
    Load a pre-trained Rock Paper Scissors model.
    
    Args:
        model_path (str): Path to the saved model
        model_format (str): Format of the model ('keras', 'h5', 'savedmodel')
        
    Returns:
        keras.Model: Loaded model ready for inference
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If unsupported model format
    """
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        if model_format.lower() == 'keras' or model_path.endswith('.keras'):
            logger.info(f"Loading Keras model from: {model_path}")
            model = keras.models.load_model(model_path)
            
        elif model_format.lower() == 'h5' or model_path.endswith('.h5'):
            logger.info(f"Loading HDF5 model from: {model_path}")
            model = keras.models.load_model(model_path)
            
        elif model_format.lower() == 'savedmodel':
            logger.info(f"Loading SavedModel from: {model_path}")
            model = keras.models.load_model(model_path)
            
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
            
        logger.info("Model loaded successfully!")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def load_model_metadata(metadata_path):
    """
    Load model metadata from JSON file.
    
    Args:
        metadata_path (str): Path to the metadata JSON file
        
    Returns:
        dict: Model metadata including metrics, preprocessing info, etc.
    """
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info("Model metadata loaded successfully")
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return {}


def get_model_info(model):
    """
    Get detailed information about a loaded model.
    
    Args:
        model (keras.Model): Loaded Keras model
        
    Returns:
        dict: Dictionary containing model information
    """
    
    try:
        info = {
            'model_name': model.name,
            'total_parameters': model.count_params(),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'layers': len(model.layers),
            'trainable_parameters': sum([tf.size(var).numpy() for var in model.trainable_variables]),
            'optimizer': model.optimizer.__class__.__name__ if hasattr(model, 'optimizer') else 'Unknown',
            'loss_function': model.loss if hasattr(model, 'loss') else 'Unknown'
        }
        
        # Get layer information
        info['layer_details'] = []
        for i, layer in enumerate(model.layers):
            layer_info = {
                'index': i,
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'Unknown'
            }
            info['layer_details'].append(layer_info)
            
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {}


def validate_model_for_inference(model, expected_classes=CLASS_NAMES):
    """
    Validate that a model is ready for inference.
    
    Args:
        model (keras.Model): Model to validate
        expected_classes (list): Expected class names
        
    Returns:
        bool: True if model is valid for inference
        
    Raises:
        ValueError: If model validation fails
    """
    
    # Check model structure
    if model is None:
        raise ValueError("Model is None")
    
    # Check input shape
    expected_input_shape = (None, IMG_HEIGHT, IMG_WIDTH, 3)
    if model.input_shape != expected_input_shape:
        logger.warning(f"Input shape mismatch. Expected: {expected_input_shape}, Got: {model.input_shape}")
    
    # Check output shape
    expected_output_shape = (None, len(expected_classes))
    if model.output_shape != expected_output_shape:
        raise ValueError(f"Output shape mismatch. Expected: {expected_output_shape}, Got: {model.output_shape}")
    
    # Test with dummy input
    try:
        dummy_input = np.random.random((1, IMG_HEIGHT, IMG_WIDTH, 3))
        predictions = model.predict(dummy_input, verbose=0)
        
        if predictions.shape != (1, len(expected_classes)):
            raise ValueError(f"Prediction shape mismatch: {predictions.shape}")
            
        if not np.allclose(np.sum(predictions, axis=1), 1.0, rtol=1e-5):
            raise ValueError("Predictions do not sum to 1 (not softmax)")
            
    except Exception as e:
        raise ValueError(f"Model validation failed during inference test: {str(e)}")
    
    logger.info("Model validation passed")
    return True


# Convenience function for common use case
def load_production_model(models_dir='../models', model_name='rock_paper_scissors_model'):
    """
    Load the production model with metadata.
    
    Args:
        models_dir (str): Directory containing model files
        model_name (str): Base name of the model files
        
    Returns:
        tuple: (model, metadata) or (None, None) if loading fails
    """
    
    # Try loading in order of preference: .keras -> .h5 -> savedmodel
    model_files = [
        (f"{model_name}.keras", 'keras'),
        (f"{model_name}.h5", 'h5'),
        (f"{model_name}", 'savedmodel')
    ]
    
    for filename, format_type in model_files:
        model_path = os.path.join(models_dir, filename)
        
        if os.path.exists(model_path):
            try:
                model = load_trained_model(model_path, format_type)
                metadata_path = os.path.join(models_dir, 'model_metadata.json')
                metadata = load_model_metadata(metadata_path)
                
                # Validate model
                validate_model_for_inference(model)
                
                logger.info(f"Production model loaded successfully from: {model_path}")
                return model, metadata
                
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {str(e)}")
                continue
    
    logger.error("Failed to load any production model")
    return None, None


if __name__ == "__main__":
    # Test the model creation and loading functionality
    print("🧪 Testing Rock Paper Scissors Model Module")
    print("=" * 50)
    
    # Test model creation
    print("Creating new CNN model...")
    model = create_cnn_model()
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Test model info
    print("\nGetting model information...")
    info = get_model_info(model)
    print(f"✅ Model has {info['layers']} layers")
    
    # Test production model loading (if available)
    print("\nTesting production model loading...")
    prod_model, metadata = load_production_model()
    if prod_model is not None:
        print("✅ Production model loaded successfully!")
        if metadata:
            print(f"📊 Model accuracy: {metadata.get('metrics', {}).get('accuracy', 'N/A')}")
    else:
        print("⚠️  No production model found (this is normal for first run)")
    
    print("\n🎉 Model module testing complete!")
