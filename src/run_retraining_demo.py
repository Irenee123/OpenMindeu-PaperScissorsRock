"""
Simple Retraining Demo Script

This script demonstrates the complete retraining process:
1. Data file uploading + saving to database
2. Data preprocessing of uploaded data  
3. Retraining using custom model as pre-trained base

Run this script to see the full retraining pipeline in action.
"""

import os
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retraining import RetrainingManager

def main():
    """Main retraining demonstration"""
    
    print("🚀 ROCK PAPER SCISSORS RETRAINING DEMONSTRATION")
    print("=" * 70)
    print("This demonstrates the complete retraining pipeline:")
    print("1. ✅ Data file uploading + saving to database")
    print("2. ✅ Data preprocessing of uploaded data")
    print("3. ✅ Retraining using custom model as pre-trained base")
    print()
    
    # Initialize the retraining system
    print("🔧 Initializing retraining system...")
    manager = RetrainingManager()
    print("✅ Retraining system initialized")
    print()
    
    # Step 1: Prepare demo data (simulate user uploading new images)
    print("📤 STEP 1: Data File Uploading + Saving to Database")
    print("-" * 50)
    
    demo_images = []
    test_dir = Path("../data/test")
    
    if test_dir.exists():
        # Collect some demo images from each class
        for class_name in ["paper", "rock", "scissors"]:
            class_dir = test_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.png"))[:2]  # Take 2 images per class
                for img_path in images:
                    demo_images.append((str(img_path), class_name))
                    print(f"   📁 Found demo image: {img_path.name} -> {class_name}")
    
    print(f"   📊 Total demo images to upload: {len(demo_images)}")
    
    # Upload the demo images
    print("   🔄 Uploading images to database...")
    upload_result = manager.upload_manager.upload_batch_images(demo_images)
    
    if upload_result['successful_uploads'] > 0:
        print(f"   ✅ Successfully uploaded {upload_result['successful_uploads']} images")
        print(f"   📊 Database entries created and files saved")
    else:
        print("   ❌ No images were uploaded successfully")
        return
    
    print()
    
    # Step 2: Data Preprocessing
    print("🔄 STEP 2: Data Preprocessing of Uploaded Data")
    print("-" * 50)
    
    try:
        print("   🔄 Preprocessing uploaded images...")
        train_data, val_data = manager.preprocessor.preprocess_uploaded_data()
        print(f"   ✅ Preprocessing complete:")
        print(f"      - Training batches: {len(train_data)}")
        print(f"      - Validation batches: {len(val_data)}")
        print(f"      - Images normalized and augmented")
    except Exception as e:
        print(f"   ❌ Preprocessing failed: {e}")
        return
    
    print()
    
    # Step 3: Model Retraining
    print("🧠 STEP 3: Retraining Using Custom Model as Pre-trained Base")
    print("-" * 50)
    
    pre_trained_model_path = "../models/rock_paper_scissors_model.keras"
    
    if not os.path.exists(pre_trained_model_path):
        print(f"   ❌ Pre-trained model not found: {pre_trained_model_path}")
        print("   💡 Please run the training notebook first to create the base model")
        return
    
    print(f"   📂 Using pre-trained model: {pre_trained_model_path}")
    print("   🔄 Starting retraining process...")
    
    # Execute retraining
    retraining_result = manager.retrainer.retrain_model(
        pre_trained_model_path=pre_trained_model_path,
        new_train_data=train_data,
        new_val_data=val_data,
        epochs=5,  # Quick demo with 5 epochs
        learning_rate=0.0001  # Lower learning rate for fine-tuning
    )
    
    if retraining_result['success']:
        print("   ✅ Retraining completed successfully!")
        print(f"   📊 Retraining Results:")
        print(f"      - Session ID: {retraining_result['session_id']}")
        print(f"      - Initial accuracy: {retraining_result['initial_accuracy']:.4f}")
        print(f"      - Final accuracy: {retraining_result['final_accuracy']:.4f}")
        print(f"      - Improvement: {retraining_result['improvement']:.4f}")
        print(f"      - Training epochs: {retraining_result['training_epochs']}")
        print(f"      - New model saved: {retraining_result['retrained_model_path']}")
        
        # Mark images as processed
        unprocessed = manager.db_manager.get_unprocessed_images()
        if unprocessed:
            image_ids = [img['id'] for img in unprocessed]
            manager.db_manager.mark_images_processed(image_ids)
            print(f"   📝 Marked {len(image_ids)} images as processed in database")
        
    else:
        print(f"   ❌ Retraining failed: {retraining_result['error']}")
        return
    
    print()
    print("🎉 RETRAINING DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("✅ All required components demonstrated:")
    print("   1. ✓ Data file uploading + saving to database")
    print("   2. ✓ Data preprocessing of uploaded data")
    print("   3. ✓ Retraining using custom model as pre-trained base")
    print()
    print("📁 Files created:")
    print(f"   - Database: ../models/retraining.db")
    print(f"   - Retrained model: {retraining_result.get('retrained_model_path', 'N/A')}")
    print(f"   - Metadata: {retraining_result.get('metadata_path', 'N/A')}")
    print()
    print("🚀 Your retraining system is fully functional and ready for evaluation!")

if __name__ == "__main__":
    main()
