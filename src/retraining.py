"""
Rock Paper Scissors Model Retraining System

This module handles the complete retraining pipeline including:
1. Data file uploading and saving to database
2. Data preprocessing of uploaded data
3. Retraining using the existing model as pre-trained base

"""

import os
import sys
import json
import sqlite3
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import zipfile
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import load_production_model, create_cnn_model, CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH
from preprocessing import get_train_test_data
from prediction import RockPaperScissorsPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrainingDatabase:
    """Database manager for storing uploaded data and retraining metadata"""
    
    def __init__(self, db_path='../models/retraining.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the retraining database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for uploaded images
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploaded_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                class_label TEXT NOT NULL,
                file_path TEXT NOT NULL,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Table for retraining sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retraining_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                images_count INTEGER,
                pre_trained_model_path TEXT,
                new_model_path TEXT,
                training_epochs INTEGER,
                initial_accuracy REAL,
                final_accuracy REAL,
                status TEXT DEFAULT 'started'
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Retraining database initialized successfully")
    
    def save_uploaded_image(self, filename: str, class_label: str, file_path: str, file_size: int) -> int:
        """Save uploaded image metadata to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO uploaded_images (filename, class_label, file_path, file_size)
            VALUES (?, ?, ?, ?)
        ''', (filename, class_label, file_path, file_size))
        
        image_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Saved uploaded image: {filename} (ID: {image_id})")
        return image_id
    
    def get_unprocessed_images(self) -> List[Dict]:
        """Get all unprocessed uploaded images"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, class_label, file_path, upload_timestamp, file_size
            FROM uploaded_images 
            WHERE processed = FALSE
            ORDER BY upload_timestamp
        ''')
        
        images = []
        for row in cursor.fetchall():
            images.append({
                'id': row[0],
                'filename': row[1],
                'class_label': row[2],
                'file_path': row[3],
                'upload_timestamp': row[4],
                'file_size': row[5]
            })
        
        conn.close()
        return images
    
    def mark_images_processed(self, image_ids: List[int]):
        """Mark images as processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(image_ids))
        cursor.execute(f'''
            UPDATE uploaded_images 
            SET processed = TRUE 
            WHERE id IN ({placeholders})
        ''', image_ids)
        
        conn.commit()
        conn.close()
        logger.info(f"Marked {len(image_ids)} images as processed")
    
    def create_retraining_session(self, images_count: int, pre_trained_model_path: str) -> int:
        """Create a new retraining session record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO retraining_sessions (images_count, pre_trained_model_path)
            VALUES (?, ?)
        ''', (images_count, pre_trained_model_path))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Created retraining session: {session_id}")
        return session_id
    
    def update_retraining_session(self, session_id: int, **kwargs):
        """Update retraining session with results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        values = []
        for key, value in kwargs.items():
            updates.append(f"{key} = ?")
            values.append(value)
        
        values.append(session_id)
        update_query = f"UPDATE retraining_sessions SET {', '.join(updates)} WHERE id = ?"
        
        cursor.execute(update_query, values)
        conn.commit()
        conn.close()
        
        logger.info(f"Updated retraining session {session_id}")

class DataUploadManager:
    """Handles data file uploading and saving for retraining"""
    
    def __init__(self, upload_dir='../data/uploads', db_manager: RetrainingDatabase = None):
        self.upload_dir = upload_dir
        self.db_manager = db_manager or RetrainingDatabase()
        os.makedirs(upload_dir, exist_ok=True)
        
        # Create class subdirectories
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(upload_dir, class_name), exist_ok=True)
    
    def upload_single_image(self, image_file_path: str, class_label: str) -> Dict:
        """
        Upload and save a single image file
        
        Args:
            image_file_path: Path to the image file to upload
            class_label: Class label (paper, rock, or scissors)
        
        Returns:
            Dictionary with upload result
        """
        try:
            if class_label not in CLASS_NAMES:
                raise ValueError(f"Invalid class label: {class_label}. Must be one of {CLASS_NAMES}")
            
            if not os.path.exists(image_file_path):
                raise FileNotFoundError(f"Image file not found: {image_file_path}")
            
            # Validate image
            try:
                img = Image.open(image_file_path)
                img.verify()  # Verify it's a valid image
            except Exception as e:
                raise ValueError(f"Invalid image file: {e}")
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{os.path.basename(image_file_path)}"
            
            # Save to class directory
            destination_path = os.path.join(self.upload_dir, class_label, filename)
            shutil.copy2(image_file_path, destination_path)
            
            # Get file size
            file_size = os.path.getsize(destination_path)
            
            # Save to database
            image_id = self.db_manager.save_uploaded_image(
                filename=filename,
                class_label=class_label,
                file_path=destination_path,
                file_size=file_size
            )
            
            logger.info(f"Successfully uploaded image: {filename} -> {class_label}")
            
            return {
                'success': True,
                'image_id': image_id,
                'filename': filename,
                'class_label': class_label,
                'file_path': destination_path,
                'file_size': file_size
            }
            
        except Exception as e:
            logger.error(f"Failed to upload image {image_file_path}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def upload_batch_images(self, image_files: List[Tuple[str, str]]) -> Dict:
        """
        Upload multiple images in batch
        
        Args:
            image_files: List of (image_path, class_label) tuples
        
        Returns:
            Dictionary with batch upload results
        """
        results = []
        successful_uploads = 0
        failed_uploads = 0
        
        for image_path, class_label in image_files:
            result = self.upload_single_image(image_path, class_label)
            results.append(result)
            
            if result['success']:
                successful_uploads += 1
            else:
                failed_uploads += 1
        
        logger.info(f"Batch upload completed: {successful_uploads} successful, {failed_uploads} failed")
        
        return {
            'total_files': len(image_files),
            'successful_uploads': successful_uploads,
            'failed_uploads': failed_uploads,
            'results': results
        }
    
    def upload_from_zip(self, zip_file_path: str) -> Dict:
        """
        Upload images from a ZIP file organized by class folders
        
        Args:
            zip_file_path: Path to ZIP file containing class folders
        
        Returns:
            Dictionary with upload results
        """
        try:
            temp_extract_dir = os.path.join(self.upload_dir, 'temp_extract')
            os.makedirs(temp_extract_dir, exist_ok=True)
            
            # Extract ZIP file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            
            # Find images in extracted folders
            image_files = []
            for class_name in CLASS_NAMES:
                class_dir = os.path.join(temp_extract_dir, class_name)
                if os.path.exists(class_dir):
                    for file in os.listdir(class_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(class_dir, file)
                            image_files.append((image_path, class_name))
            
            # Upload all found images
            results = self.upload_batch_images(image_files)
            
            # Clean up temp directory
            shutil.rmtree(temp_extract_dir)
            
            logger.info(f"ZIP upload completed: {len(image_files)} images processed")
            return results
            
        except Exception as e:
            logger.error(f"Failed to upload from ZIP {zip_file_path}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class RetrainingPreprocessor:
    """Handles preprocessing of uploaded data for retraining"""
    
    def __init__(self, upload_dir='../data/uploads'):
        self.upload_dir = upload_dir
    
    def preprocess_uploaded_data(self, validation_split=0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Preprocess uploaded data for retraining
        
        Args:
            validation_split: Fraction of data to use for validation
        
        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        try:
            # Use the same preprocessing as original training
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            # Create data generators for uploaded data
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=validation_split
            )
            
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
            
            # Load training subset
            train_data = train_datagen.flow_from_directory(
                self.upload_dir,
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=16,  # Smaller batch for retraining
                class_mode='categorical',
                subset='training'
            )
            
            # Load validation subset
            val_data = val_datagen.flow_from_directory(
                self.upload_dir,
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=16,
                class_mode='categorical',
                subset='validation'
            )
            
            logger.info(f"Preprocessed uploaded data: {len(train_data)} train batches, {len(val_data)} val batches")
            
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"Failed to preprocess uploaded data: {e}")
            raise

class ModelRetrainer:
    """Handles model retraining using pre-trained model as base"""
    
    def __init__(self, db_manager: RetrainingDatabase = None):
        self.db_manager = db_manager or RetrainingDatabase()
    
    def retrain_model(self, 
                     pre_trained_model_path: str,
                     new_train_data: tf.data.Dataset,
                     new_val_data: tf.data.Dataset,
                     epochs: int = 10,
                     learning_rate: float = 0.0001) -> Dict:
        """
        Retrain model using uploaded data with pre-trained model as base
        
        Args:
            pre_trained_model_path: Path to the pre-trained model
            new_train_data: New training data
            new_val_data: New validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
        
        Returns:
            Dictionary with retraining results
        """
        try:
            # Create retraining session
            session_id = self.db_manager.create_retraining_session(
                images_count=len(list(new_train_data)) * 32,  # Approximate
                pre_trained_model_path=pre_trained_model_path
            )
            
            logger.info(f"Starting retraining session {session_id}")
            
            # Load pre-trained model
            logger.info(f"Loading pre-trained model from: {pre_trained_model_path}")
            base_model = keras.models.load_model(pre_trained_model_path)
            
            # Get initial accuracy on validation data
            initial_results = base_model.evaluate(new_val_data, verbose=0)
            initial_accuracy = initial_results[1] if len(initial_results) > 1 else 0.0
            
            logger.info(f"Initial model accuracy on new data: {initial_accuracy:.4f}")
            
            # Fine-tune the model with lower learning rate
            base_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Setup callbacks for retraining
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            retrained_model_path = f"../models/retrained_model_{timestamp}.keras"
            
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    retrained_model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Retrain the model
            logger.info(f"Starting retraining for {epochs} epochs...")
            history = base_model.fit(
                new_train_data,
                epochs=epochs,
                validation_data=new_val_data,
                callbacks=callbacks,
                verbose=1
            )
            
            # Get final accuracy
            final_results = base_model.evaluate(new_val_data, verbose=0)
            final_accuracy = final_results[1] if len(final_results) > 1 else 0.0
            
            logger.info(f"Final model accuracy after retraining: {final_accuracy:.4f}")
            
            # Save the retrained model
            base_model.save(retrained_model_path)
            
            # Update retraining session
            self.db_manager.update_retraining_session(
                session_id=session_id,
                new_model_path=retrained_model_path,
                training_epochs=len(history.history['loss']),
                initial_accuracy=initial_accuracy,
                final_accuracy=final_accuracy,
                status='completed'
            )
            
            # Create retraining metadata
            retraining_metadata = {
                'session_id': session_id,
                'pre_trained_model': pre_trained_model_path,
                'retrained_model': retrained_model_path,
                'training_epochs': len(history.history['loss']),
                'initial_accuracy': float(initial_accuracy),
                'final_accuracy': float(final_accuracy),
                'improvement': float(final_accuracy - initial_accuracy),
                'learning_rate': learning_rate,
                'timestamp': timestamp,
                'training_history': {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']],
                    'accuracy': [float(x) for x in history.history['accuracy']],
                    'val_accuracy': [float(x) for x in history.history['val_accuracy']]
                }
            }
            
            # Save metadata
            metadata_path = f"../models/retraining_metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(retraining_metadata, f, indent=2)
            
            logger.info(f"Retraining completed successfully! Session ID: {session_id}")
            logger.info(f"Model improvement: {final_accuracy - initial_accuracy:.4f}")
            
            return {
                'success': True,
                'session_id': session_id,
                'retrained_model_path': retrained_model_path,
                'metadata_path': metadata_path,
                'initial_accuracy': initial_accuracy,
                'final_accuracy': final_accuracy,
                'improvement': final_accuracy - initial_accuracy,
                'training_epochs': len(history.history['loss'])
            }
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            # Update session status
            if 'session_id' in locals():
                self.db_manager.update_retraining_session(
                    session_id=session_id,
                    status=f'failed: {str(e)}'
                )
            
            return {
                'success': False,
                'error': str(e)
            }

class RetrainingManager:
    """Main manager for the complete retraining process"""
    
    def __init__(self):
        self.db_manager = RetrainingDatabase()
        self.upload_manager = DataUploadManager(db_manager=self.db_manager)
        self.preprocessor = RetrainingPreprocessor()
        self.retrainer = ModelRetrainer(db_manager=self.db_manager)
    
    def full_retraining_pipeline(self, 
                               image_files: List[Tuple[str, str]] = None,
                               zip_file_path: str = None,
                               pre_trained_model_path: str = '../models/rock_paper_scissors_model.keras',
                               epochs: int = 10) -> Dict:
        """
        Execute the complete retraining pipeline
        
        Args:
            image_files: List of (image_path, class_label) tuples to upload
            zip_file_path: Path to ZIP file containing images
            pre_trained_model_path: Path to pre-trained model
            epochs: Number of retraining epochs
        
        Returns:
            Dictionary with complete pipeline results
        """
        try:
            logger.info("🚀 Starting complete retraining pipeline...")
            
            # Step 1: Data Upload and Database Storage
            logger.info("📤 Step 1: Uploading and saving data to database...")
            
            if zip_file_path:
                upload_result = self.upload_manager.upload_from_zip(zip_file_path)
            elif image_files:
                upload_result = self.upload_manager.upload_batch_images(image_files)
            else:
                # Use existing unprocessed images
                unprocessed = self.db_manager.get_unprocessed_images()
                if not unprocessed:
                    raise ValueError("No new data to process. Please upload images first.")
                upload_result = {'successful_uploads': len(unprocessed)}
            
            if upload_result.get('successful_uploads', 0) == 0:
                raise ValueError("No images were successfully uploaded")
            
            logger.info(f"✅ Successfully uploaded {upload_result.get('successful_uploads', 0)} images")
            
            # Step 2: Data Preprocessing
            logger.info("🔄 Step 2: Preprocessing uploaded data...")
            
            train_data, val_data = self.preprocessor.preprocess_uploaded_data()
            logger.info("✅ Data preprocessing completed")
            
            # Step 3: Model Retraining with Pre-trained Base
            logger.info("🧠 Step 3: Retraining model using pre-trained base...")
            
            retraining_result = self.retrainer.retrain_model(
                pre_trained_model_path=pre_trained_model_path,
                new_train_data=train_data,
                new_val_data=val_data,
                epochs=epochs
            )
            
            if not retraining_result['success']:
                raise Exception(f"Retraining failed: {retraining_result['error']}")
            
            # Mark uploaded images as processed
            unprocessed_images = self.db_manager.get_unprocessed_images()
            if unprocessed_images:
                image_ids = [img['id'] for img in unprocessed_images]
                self.db_manager.mark_images_processed(image_ids)
            
            logger.info("🎉 Complete retraining pipeline finished successfully!")
            
            return {
                'success': True,
                'upload_result': upload_result,
                'retraining_result': retraining_result,
                'pipeline_status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'pipeline_status': 'failed'
            }

# Demo script for testing retraining
def demo_retraining():
    """Demonstrate the complete retraining process"""
    
    print("🧪 ROCK PAPER SCISSORS MODEL RETRAINING DEMO")
    print("=" * 60)
    
    # Initialize retraining manager
    manager = RetrainingManager()
    
    # Demo 1: Upload some sample images (simulate new data)
    print("\n📤 Demo 1: Simulating data upload...")
    
    # Create some demo upload data (using existing test images)
    demo_uploads = []
    test_data_dir = "../data/test"
    
    if os.path.exists(test_data_dir):
        for class_name in CLASS_NAMES[:2]:  # Just use 2 classes for demo
            class_dir = os.path.join(test_data_dir, class_name)
            if os.path.exists(class_dir):
                images = os.listdir(class_dir)[:3]  # Take first 3 images
                for img in images:
                    img_path = os.path.join(class_dir, img)
                    demo_uploads.append((img_path, class_name))
    
    print(f"Found {len(demo_uploads)} demo images to upload")
    
    # Demo 2: Execute full retraining pipeline
    print("\n🚀 Demo 2: Executing full retraining pipeline...")
    
    result = manager.full_retraining_pipeline(
        image_files=demo_uploads,
        epochs=3  # Quick demo with few epochs
    )
    
    if result['success']:
        print("✅ Retraining pipeline completed successfully!")
        print(f"📊 Results:")
        retraining_info = result['retraining_result']
        print(f"   - Session ID: {retraining_info['session_id']}")
        print(f"   - Initial Accuracy: {retraining_info['initial_accuracy']:.4f}")
        print(f"   - Final Accuracy: {retraining_info['final_accuracy']:.4f}")
        print(f"   - Improvement: {retraining_info['improvement']:.4f}")
        print(f"   - Training Epochs: {retraining_info['training_epochs']}")
        print(f"   - Retrained Model: {retraining_info['retrained_model_path']}")
    else:
        print(f"❌ Retraining pipeline failed: {result['error']}")
    
    print("\n🎯 RETRAINING DEMO COMPLETE!")

if __name__ == "__main__":
    demo_retraining()
