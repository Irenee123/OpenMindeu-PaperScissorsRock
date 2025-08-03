#!/usr/bin/env python3
"""
Test script for Rock Paper Scissors prediction system
"""

import sys
import os
sys.path.append('src')

from prediction import RockPaperScissorsPredictor

def main():
    print('🚀 Testing Rock Paper Scissors Prediction System')
    print('=' * 60)
    
    # Initialize predictor
    print('📝 Initializing predictor...')
    try:
        predictor = RockPaperScissorsPredictor()
        print('✅ Predictor initialized successfully!')
    except Exception as e:
        print(f'❌ Error initializing predictor: {e}')
        return
    
    # Test with a sample image
    print('\n🔍 Testing single image prediction...')
    test_image_path = 'data/test/rock/00Aw9Tyn4eIYJ21y.png'
    
    if os.path.exists(test_image_path):
        print(f'📸 Using test image: {test_image_path}')
        try:
            result = predictor.predict_with_action(test_image_path)
            print('✅ Prediction successful!')
            print(f'   📸 Image: {os.path.basename(test_image_path)}')
            print(f'   🎯 Predicted: {result["predicted_class"]}')
            print(f'   📊 Confidence: {result["confidence"]:.4f}')
            print(f'   🚦 Action: {result["action"]}')
            print(f'   💭 Reasoning: {result["reasoning"]}')
        except Exception as e:
            print(f'❌ Error making prediction: {e}')
    else:
        print(f'⚠️ Test image not found: {test_image_path}')
        print('🔄 Looking for alternative test images...')
        
        # Try to find any test image
        for class_name in ['rock', 'paper', 'scissors']:
            test_dir = f'data/test/{class_name}'
            if os.path.exists(test_dir):
                files = os.listdir(test_dir)
                if files:
                    test_image_path = os.path.join(test_dir, files[0])
                    print(f'✅ Using alternative: {test_image_path}')
                    try:
                        result = predictor.predict_with_action(test_image_path)
                        print('✅ Prediction successful!')
                        print(f'   📸 Image: {os.path.basename(test_image_path)}')
                        print(f'   🎯 Predicted: {result["predicted_class"]}')
                        print(f'   📊 Confidence: {result["confidence"]:.4f}')
                        print(f'   🚦 Action: {result["action"]}')
                        print(f'   💭 Reasoning: {result["reasoning"]}')
                        break
                    except Exception as e:
                        print(f'❌ Error making prediction: {e}')
                        continue
    
    # Test batch prediction if we found an image
    print('\n🔍 Testing batch prediction...')
    try:
        # Find a few test images
        test_images = []
        for class_name in ['rock', 'paper', 'scissors']:
            test_dir = f'data/test/{class_name}'
            if os.path.exists(test_dir):
                files = os.listdir(test_dir)[:2]  # Get first 2 images
                for file in files:
                    test_images.append(os.path.join(test_dir, file))
        
        if test_images:
            results = predictor.batch_predict_with_actions(test_images[:3])  # Test with 3 images
            print(f'✅ Batch prediction successful! Processed {len(results)} images')
            for i, result in enumerate(results):
                if 'error' not in result:
                    print(f'   Image {i+1}: {result["predicted_class"]} (conf: {result["confidence"]:.3f}, action: {result["action"]})')
        else:
            print('⚠️ No test images found for batch prediction')
            
    except Exception as e:
        print(f'❌ Error in batch prediction: {e}')
    
    # Show statistics
    print('\n📊 Prediction Statistics:')
    try:
        stats = predictor.get_prediction_statistics()
        print(f'   Total predictions: {stats["total_predictions"]}')
        print(f'   Accept: {stats["accept"]} ({stats["accept_rate"]:.1%})')
        print(f'   Review: {stats["review"]} ({stats["review_rate"]:.1%})')
        print(f'   Reject: {stats["reject"]} ({stats["reject_rate"]:.1%})')
        print(f'   Average confidence: {stats["average_confidence"]:.3f}')
    except Exception as e:
        print(f'❌ Error getting statistics: {e}')
    
    print('\n🎉 Prediction system test completed!')

if __name__ == '__main__':
    main()
