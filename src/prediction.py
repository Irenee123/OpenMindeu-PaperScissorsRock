"""
Rock Paper Scissors Prediction Module

This module handles image preprocessing and prediction for the Rock Paper Scissors 
classifier. It provides functions for single image prediction, batch prediction,
and confidence analysis. This is part of the ML summative assignment.

Author: ML Student
Date: August 2025
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import logging
from typing import Union, List, Dict, Tuple, Optional
import json
import time
from enum import Enum

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import load_production_model, CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH

class ConfidenceAction(Enum):
    """Enum for confidence-based actions"""
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RockPaperScissorsPredictor:
    """
    A comprehensive predictor class for Rock Paper Scissors classification.
    
    This class handles model loading, image preprocessing, prediction, and
    confidence analysis for the Rock Paper Scissors CNN model with
    actionable confidence-based decisions.
    """
    
    def __init__(self, models_dir='../models', model_name='rock_paper_scissors_model',
                 confidence_thresholds: Dict[str, float] = None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            models_dir (str): Directory containing model files
            model_name (str): Base name of the model files
            confidence_thresholds (dict): Custom confidence thresholds for actions
        """
        self.models_dir = models_dir
        self.model_name = model_name
        self.model = None
        self.metadata = None
        self.class_names = CLASS_NAMES
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH
        
        # Set confidence thresholds for actionable decisions
        self.confidence_thresholds = confidence_thresholds or {
            'accept': 0.85,    # Accept prediction if confidence >= 85%
            'review': 0.60,    # Manual review if 60% <= confidence < 85%
            'reject': 0.60     # Reject prediction if confidence < 60%
        }
        
        # Statistics tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'accept': 0,
            'review': 0,
            'reject': 0,
            'avg_confidence': 0.0
        }
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and metadata."""
        try:
            self.model, self.metadata = load_production_model(
                models_dir=self.models_dir,
                model_name=self.model_name
            )
            
            if self.model is None:
                raise ValueError("Failed to load production model")
                
            # Update class names and image dimensions from metadata if available
            if self.metadata:
                self.class_names = self.metadata.get('classes', CLASS_NAMES)
                preprocessing_info = self.metadata.get('preprocessing', {})
                target_size = preprocessing_info.get('target_size', [IMG_HEIGHT, IMG_WIDTH])
                self.img_height, self.img_width = target_size
                
            logger.info("Predictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {str(e)}")
            raise
    
    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess an image for prediction.
        
        Args:
            image_input: Can be a file path (str), numpy array, or PIL Image
            
        Returns:
            np.ndarray: Preprocessed image ready for prediction
            
        Raises:
            ValueError: If image processing fails
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                img = keras.preprocessing.image.load_img(
                    image_input, 
                    target_size=(self.img_height, self.img_width)
                )
                
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                img = Image.fromarray(image_input.astype('uint8'))
                img = img.resize((self.img_width, self.img_height))
                
            elif isinstance(image_input, Image.Image):
                # PIL Image
                img = image_input.resize((self.img_width, self.img_height))
                
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Convert to array and normalize
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Normalize to [0,1]
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def predict_single(self, image_input: Union[str, np.ndarray, Image.Image], 
                      return_confidence: bool = True, actionable: bool = False) -> Dict:
        """
        Predict the class of a single image with optional actionable confidence.
        
        Args:
            image_input: Image to predict (file path, numpy array, or PIL Image)
            return_confidence: Whether to return confidence scores for all classes
            actionable: Whether to include actionable confidence decisions
            
        Returns:
            dict: Prediction results with optional actionable confidence
        """
        try:
            start_time = time.time()
            
            # Preprocess image
            processed_image = self.preprocess_image(image_input)
            
            # Make prediction using the real trained model
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence_scores = predictions[0]
            
            # Prepare results
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(confidence_scores[predicted_class_idx])
            
            prediction_time = time.time() - start_time
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'prediction_time': prediction_time,
                'model_version': self.metadata.get('version', '1.0') if self.metadata else '1.0'
            }
            
            if return_confidence:
                result['all_probabilities'] = {
                    self.class_names[i]: float(confidence_scores[i]) 
                    for i in range(len(self.class_names))
                }
                
                # Add confidence level interpretation
                if confidence >= 0.9:
                    result['confidence_level'] = 'Very High'
                elif confidence >= 0.75:
                    result['confidence_level'] = 'High'
                elif confidence >= 0.6:
                    result['confidence_level'] = 'Medium'
                else:
                    result['confidence_level'] = 'Low'
            
            # Add actionable confidence decisions
            if actionable:
                action_result = self._determine_action(confidence, confidence_scores)
                result.update(action_result)
                
                # Update statistics
                self._update_stats(confidence, action_result['action'])
            
            logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'error': str(e),
                'predicted_class': None,
                'confidence': 0.0
            }
    
    def predict_batch(self, image_inputs: List[Union[str, np.ndarray, Image.Image]], 
                     batch_size: int = 32) -> List[Dict]:
        """
        Predict classes for multiple images efficiently.
        
        Args:
            image_inputs: List of images to predict
            batch_size: Batch size for processing
            
        Returns:
            list: List of prediction results for each image
        """
        try:
            results = []
            total_images = len(image_inputs)
            
            logger.info(f"Processing batch of {total_images} images")
            
            for i in range(0, total_images, batch_size):
                batch_inputs = image_inputs[i:i+batch_size]
                batch_results = []
                
                # Process each image in the batch
                for img_input in batch_inputs:
                    try:
                        result = self.predict_single(img_input, return_confidence=False)
                        batch_results.append(result)
                    except Exception as e:
                        batch_results.append({
                            'error': str(e),
                            'predicted_class': None,
                            'confidence': 0.0
                        })
                
                results.extend(batch_results)
                logger.info(f"Processed {min(i+batch_size, total_images)}/{total_images} images")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            return [{'error': str(e)} for _ in image_inputs]
    
    def analyze_prediction_confidence(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Perform detailed confidence analysis on a prediction.
        
        Args:
            image_input: Image to analyze
            
        Returns:
            dict: Detailed confidence analysis
        """
        try:
            result = self.predict_single(image_input, return_confidence=True)
            
            if 'error' in result:
                return result
            
            probabilities = result['all_probabilities']
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            analysis = {
                'prediction_summary': {
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'confidence_level': result['confidence_level']
                },
                'ranking': [
                    {'class': class_name, 'probability': prob, 'percentage': f"{prob*100:.2f}%"}
                    for class_name, prob in sorted_probs
                ],
                'confidence_analysis': {
                    'is_confident': result['confidence'] >= 0.75,
                    'uncertainty_score': 1 - result['confidence'],
                    'top_2_difference': sorted_probs[0][1] - sorted_probs[1][1],
                    'entropy': -sum([p * np.log(p + 1e-10) for p in probabilities.values()])
                },
                'recommendation': self._get_prediction_recommendation(result['confidence'], sorted_probs)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during confidence analysis: {str(e)}")
            return {'error': str(e)}
    
    def _get_prediction_recommendation(self, confidence: float, sorted_probs: List) -> str:
        """Get recommendation based on prediction confidence."""
        if confidence >= 0.9:
            return "Very confident prediction. Safe to use."
        elif confidence >= 0.75:
            return "Good confidence level. Reliable prediction."
        elif confidence >= 0.6:
            top_2_diff = sorted_probs[0][1] - sorted_probs[1][1]
            if top_2_diff > 0.2:
                return "Moderate confidence. Consider additional validation."
            else:
                return "Low confidence with close alternatives. Manual review recommended."
        else:
            return "Low confidence prediction. Manual verification strongly recommended."
    
    def _determine_action(self, confidence: float, confidence_scores: np.ndarray) -> Dict:
        """
        Determine actionable decision based on confidence levels.
        
        Args:
            confidence (float): Primary prediction confidence
            confidence_scores (np.ndarray): All class probabilities
            
        Returns:
            dict: Action decision with reasoning
        """
        # Determine action based on confidence thresholds
        if confidence >= self.confidence_thresholds['accept']:
            action = ConfidenceAction.ACCEPT
            reasoning = f"High confidence ({confidence:.1%}) - prediction accepted automatically"
            risk_level = "Low"
            
        elif confidence >= self.confidence_thresholds['review']:
            action = ConfidenceAction.REVIEW
            reasoning = f"Moderate confidence ({confidence:.1%}) - manual review recommended"
            risk_level = "Medium"
            
        else:
            action = ConfidenceAction.REJECT
            reasoning = f"Low confidence ({confidence:.1%}) - prediction rejected"
            risk_level = "High"
        
        # Calculate additional risk factors
        sorted_probs = np.sort(confidence_scores)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]  # Difference between top 2 predictions
        
        # Adjust action based on margin
        if action == ConfidenceAction.REVIEW and margin < 0.15:
            action = ConfidenceAction.REJECT
            reasoning += " (low margin between top predictions)"
            risk_level = "High"
        
        return {
            'action': action.value,
            'reasoning': reasoning,
            'risk_level': risk_level,
            'margin': float(margin),
            'thresholds_used': self.confidence_thresholds,
            'actionable': True
        }
    
    def _update_stats(self, confidence: float, action: str):
        """Update prediction statistics."""
        self.prediction_stats['total_predictions'] += 1
        self.prediction_stats[action] += 1
        
        # Update running average confidence
        total = self.prediction_stats['total_predictions']
        current_avg = self.prediction_stats['avg_confidence']
        self.prediction_stats['avg_confidence'] = ((current_avg * (total - 1)) + confidence) / total
    
    def predict_with_action(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Predict with actionable confidence decisions using your trained model.
        
        Args:
            image_input: Image to predict
            
        Returns:
            dict: Prediction with actionable decision
        """
        return self.predict_single(image_input, return_confidence=True, actionable=True)
    
    def batch_predict_with_actions(self, image_inputs: List[Union[str, np.ndarray, Image.Image]]) -> Dict:
        """
        Batch prediction with actionable confidence analysis.
        
        Args:
            image_inputs: List of images to predict
            
        Returns:
            dict: Batch results with statistics and actionable decisions
        """
        results = []
        actions_summary = {'accept': 0, 'review': 0, 'reject': 0}
        
        logger.info(f"Processing batch of {len(image_inputs)} images with actionable confidence...")
        
        for i, image_input in enumerate(image_inputs):
            try:
                result = self.predict_with_action(image_input)
                results.append(result)
                
                if 'action' in result:
                    actions_summary[result['action']] += 1
                    
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_inputs)} images")
                    
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                results.append({'error': str(e), 'action': 'reject'})
                actions_summary['reject'] += 1
        
        total_images = len(image_inputs)
        batch_stats = {
            'total_images': total_images,
            'accept_rate': actions_summary['accept'] / total_images * 100,
            'review_rate': actions_summary['review'] / total_images * 100,
            'reject_rate': actions_summary['reject'] / total_images * 100,
            'actions_summary': actions_summary
        }
        
        return {
            'predictions': results,
            'batch_statistics': batch_stats,
            'overall_stats': self.get_prediction_statistics()
        }
    
    def get_prediction_statistics(self) -> Dict:
        """Get current prediction statistics."""
        total = self.prediction_stats['total_predictions']
        if total == 0:
            return {'message': 'No predictions made yet'}
        
        return {
            'total_predictions': total,
            'acceptance_rate': (self.prediction_stats['accept'] / total) * 100,
            'review_rate': (self.prediction_stats['review'] / total) * 100,
            'rejection_rate': (self.prediction_stats['reject'] / total) * 100,
            'average_confidence': self.prediction_stats['avg_confidence'],
            'confidence_thresholds': self.confidence_thresholds
        }
    
    def update_confidence_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Update confidence thresholds for actionable decisions.
        
        Args:
            new_thresholds (dict): New threshold values
        """
        self.confidence_thresholds.update(new_thresholds)
        logger.info(f"Updated confidence thresholds: {self.confidence_thresholds}")
    
    def reset_statistics(self):
        """Reset prediction statistics."""
        self.prediction_stats = {
            'total_predictions': 0,
            'accept': 0,
            'review': 0,
            'reject': 0,
            'avg_confidence': 0.0
        }
        logger.info("Prediction statistics reset")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information and metadata
        """
        info = {
            'model_loaded': self.model is not None,
            'class_names': self.class_names,
            'input_shape': (self.img_height, self.img_width, 3),
            'num_classes': len(self.class_names)
        }
        
        if self.metadata:
            info.update({
                'model_name': self.metadata.get('model_name', 'Unknown'),
                'version': self.metadata.get('version', 'Unknown'),
                'training_accuracy': self.metadata.get('metrics', {}).get('accuracy', 'Unknown'),
                'training_epochs': self.metadata.get('training_epochs', 'Unknown'),
                'total_parameters': self.metadata.get('total_parameters', 'Unknown')
            })
        
        return info


# Convenience functions for direct usage
_global_predictor = None

def get_predictor(models_dir='../models', model_name='rock_paper_scissors_model') -> RockPaperScissorsPredictor:
    """
    Get a global predictor instance (singleton pattern).
    
    Args:
        models_dir (str): Directory containing model files
        model_name (str): Base name of the model files
        
    Returns:
        RockPaperScissorsPredictor: Global predictor instance
    """
    global _global_predictor
    
    if _global_predictor is None:
        _global_predictor = RockPaperScissorsPredictor(models_dir, model_name)
    
    return _global_predictor


def predict_image(image_input: Union[str, np.ndarray, Image.Image], 
                 models_dir='../models', model_name='rock_paper_scissors_model') -> Dict:
    """
    Quick prediction function for single images.
    
    Args:
        image_input: Image to predict
        models_dir: Directory containing model files
        model_name: Base name of the model files
        
    Returns:
        dict: Prediction results
    """
    predictor = get_predictor(models_dir, model_name)
    return predictor.predict_single(image_input)


def predict_images(image_inputs: List[Union[str, np.ndarray, Image.Image]],
                  models_dir='../models', model_name='rock_paper_scissors_model') -> List[Dict]:
    """
    Quick prediction function for multiple images.
    
    Args:
        image_inputs: List of images to predict
        models_dir: Directory containing model files
        model_name: Base name of the model files
        
    Returns:
        list: List of prediction results
    """
    predictor = get_predictor(models_dir, model_name)
    return predictor.predict_batch(image_inputs)


def analyze_image_prediction(image_input: Union[str, np.ndarray, Image.Image],
                           models_dir='../models', model_name='rock_paper_scissors_model') -> Dict:
    """
    Detailed confidence analysis for a single image.
    
    Args:
        image_input: Image to analyze
        models_dir: Directory containing model files
        model_name: Base name of the model files
        
    Returns:
        dict: Detailed analysis results
    """
    predictor = get_predictor(models_dir, model_name)
    return predictor.analyze_prediction_confidence(image_input)


if __name__ == "__main__":
    # Test the prediction module with actionable confidence
    print("🧪 Testing Rock Paper Scissors Prediction Module with Actionable Confidence")
    print("=" * 80)
    
    try:
        # Test predictor initialization
        print("Initializing predictor with actionable confidence...")
        predictor = RockPaperScissorsPredictor()
        print("✅ Predictor initialized successfully")
        
        # Show model info
        print("\nModel Information:")
        info = predictor.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print(f"\nConfidence Thresholds:")
        for action, threshold in predictor.confidence_thresholds.items():
            print(f"  {action.capitalize()}: {threshold:.1%}")
        
        # Test with sample images if available
        test_data_dir = '../data/test'
        if os.path.exists(test_data_dir):
            print(f"\nTesting actionable confidence with sample images from {test_data_dir}...")
            
            test_images = []
            for class_name in ['paper', 'rock', 'scissors']:
                class_dir = os.path.join(test_data_dir, class_name)
                if os.path.exists(class_dir):
                    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if image_files:
                        # Test first image from each class
                        test_image = os.path.join(class_dir, image_files[0])
                        test_images.append(test_image)
                        
                        print(f"\n🔍 Testing {class_name} image: {os.path.basename(test_image)}")
                        
                        # Test actionable prediction
                        result = predictor.predict_with_action(test_image)
                        if 'error' not in result:
                            print(f"  Prediction: {result['predicted_class']}")
                            print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']:.1%})")
                            print(f"  Action: {result['action'].upper()}")
                            print(f"  Risk Level: {result['risk_level']}")
                            print(f"  Reasoning: {result['reasoning']}")
                            print(f"  Margin: {result['margin']:.4f}")
                        else:
                            print(f"  Error: {result['error']}")
            
            # Batch testing if we have multiple images
            if len(test_images) > 1:
                print(f"\n📊 Batch Testing with {len(test_images)} images...")
                batch_results = predictor.batch_predict_with_actions(test_images)
                
                print("\nBatch Statistics:")
                stats = batch_results['batch_statistics']
                print(f"  Total Images: {stats['total_images']}")
                print(f"  Accept Rate: {stats['accept_rate']:.1f}%")
                print(f"  Review Rate: {stats['review_rate']:.1f}%")
                print(f"  Reject Rate: {stats['reject_rate']:.1f}%")
                
                print("\nAction Distribution:")
                for action, count in stats['actions_summary'].items():
                    print(f"  {action.capitalize()}: {count} images")
        
        else:
            print(f"⚠️  Test data directory not found: {test_data_dir}")
            print("Creating dummy test with actionable confidence...")
            
            # Test with random image
            dummy_image = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
            result = predictor.predict_with_action(dummy_image)
            print(f"Dummy prediction: {result.get('predicted_class', 'Error')}")
            print(f"Action: {result.get('action', 'Error')}")
        
        # Show final statistics
        print("\n📈 Final Statistics:")
        final_stats = predictor.get_prediction_statistics()
        if 'message' not in final_stats:
            print(f"  Total Predictions: {final_stats['total_predictions']}")
            print(f"  Acceptance Rate: {final_stats['acceptance_rate']:.1f}%")
            print(f"  Review Rate: {final_stats['review_rate']:.1f}%")
            print(f"  Rejection Rate: {final_stats['rejection_rate']:.1f}%")
            print(f"  Average Confidence: {final_stats['average_confidence']:.4f}")
        
        print("\n🎉 Actionable confidence testing complete!")
        print("✅ Your prediction.py now includes:")
        print("   • Real model predictions (not random)")
        print("   • Actionable confidence levels")
        print("   • Accept/Review/Reject decisions")
        print("   • Risk assessment")
        print("   • Batch processing with statistics")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("Make sure the trained model files are available in ../models/")
