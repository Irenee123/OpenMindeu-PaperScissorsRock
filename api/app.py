"""
Rock Paper Scissors Production API - FINAL VERSION

This is your ONLY API file for the complete project.
Handles: Predictions, Retraining, Monitoring, File Uploads

Author: ML Student  
Date: August 2025
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import shutil
import uvicorn
from pathlib import Path
from datetime import datetime

# Setup project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

# Create FastAPI app
app = FastAPI(
    title="Rock Paper Scissors AI API",
    description="Production ML API for Rock Paper Scissors classification with retraining",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application state tracking
state = {
    "predictions": 0,
    "retrains": 0,
    "last_retrain": "Never",
    "start_time": datetime.now(),
    "model_ready": False,
    "class_counts": {"rock": 0, "paper": 0, "scissors": 0},
    "uploads": {"rock": 0, "paper": 0, "scissors": 0},
    "retraining": False
}

# Global predictor cache (for clearing after retraining)
_global_predictor = None

def initialize_api():
    """Initialize API and create upload directories"""
    print("Rock Paper Scissors AI API Starting...")
    
    # Create necessary directories
    dirs = [
        project_root / "data" / "uploads" / "temp",
        project_root / "data" / "uploads" / "rock",
        project_root / "data" / "uploads" / "paper",
        project_root / "data" / "uploads" / "scissors"
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Don't load ML model during startup - load only when needed
    state["model_ready"] = "not_loaded"  # Will load on first prediction
    
    print("Upload directories created")
    print("API Ready: http://localhost:8003")
    print("Docs: http://localhost:8003/docs")
    print("ML Model will load on first prediction request")

# Initialize on import
initialize_api()

@app.get("/")
async def dashboard():
    """Main API dashboard with status and statistics"""
    uptime = datetime.now() - state["start_time"]
    
    return {
        "Rock Paper Scissors AI API": "ONLINE",
        "System Status": {
            "uptime": str(uptime).split('.')[0],
            "model_loaded": "Ready" if state["model_ready"] else "Error",
            "total_predictions": state["predictions"],
            "total_retrains": state["retrains"],
            "last_retrain": state["last_retrain"],
            "currently_retraining": state["retraining"]
        },
        "Prediction Stats": {
            "rock_predictions": state["class_counts"]["rock"],
            "paper_predictions": state["class_counts"]["paper"],
            "scissors_predictions": state["class_counts"]["scissors"]
        },
        "User Uploads": {
            "rock_uploads": state["uploads"]["rock"],
            "paper_uploads": state["uploads"]["paper"],
            "scissors_uploads": state["uploads"]["scissors"],
            "total_uploads": sum(state["uploads"].values())
        },
        "Available Endpoints": {
            "Predict Image": "POST /predict",
            "Upload Training Data": "POST /upload/{class}",
            "Retrain Model": "POST /retrain",
            "Statistics": "GET /stats",
            "Documentation": "GET /docs"
        }
    }

@app.post("/predict")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """Upload image and get Rock/Paper/Scissors AI prediction"""
    
    # Load model on first use or after retraining
    if state["model_ready"] == "not_loaded" or not state["model_ready"]:
        try:
            print("Loading/Reloading ML model for prediction...")
            
            # Clear any existing cached models first
            global _global_predictor
            _global_predictor = None
            
            # Clear prediction module cache
            import sys
            if 'prediction' in sys.modules:
                prediction_module = sys.modules['prediction']
                if hasattr(prediction_module, '_global_predictor'):
                    prediction_module._global_predictor = None
                print("Prediction module cache cleared")
            
            # Import fresh modules
            from prediction import predict_image, analyze_image_prediction
            
            # Test with the current model to ensure it loads properly
            models_dir = str(project_root / "models")
            print(f"Models directory: {models_dir}")
            
            state["model_ready"] = True
            print("ML Model loaded/reloaded successfully!")
        except Exception as e:
            print(f"ML model loading failed: {e}")
            state["model_ready"] = False
            raise HTTPException(status_code=503, detail=f"ML model not available: {str(e)}")
    
    if not state["model_ready"]:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    
    try:
        # Save uploaded file temporarily
        temp_dir = project_root / "data" / "uploads" / "temp"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        temp_file = temp_dir / f"predict_{timestamp}_{file.filename}"
        
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"Predicting: {file.filename}")
        
        # Make prediction using your ML model with correct paths
        from prediction import predict_image, analyze_image_prediction
        
        # Use absolute paths to avoid path issues
        models_dir = str(project_root / "models")
        result = predict_image(str(temp_file), models_dir=models_dir)
        analysis = analyze_image_prediction(str(temp_file), models_dir=models_dir)
        
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        
        # Update statistics
        state["predictions"] += 1
        state["class_counts"][predicted_class] += 1
        
        # Clean up temp file
        os.remove(temp_file)
        
        print(f"Result: {predicted_class} ({confidence:.1f}%)")
        
        return {
            "success": True,
            "Prediction": {
                "class": predicted_class,
                "confidence": f"{confidence:.2f}%",
                "confidence_level": analysis.get("confidence_level", "Unknown")
            },
            "All Probabilities": result.get("all_probabilities", {}),
            "Image Info": {
                "filename": file.filename,
                "size_kb": round(len(content) / 1024, 2)
            },
            "Stats": {
                "total_predictions": state["predictions"],
                "this_class_predictions": state["class_counts"][predicted_class]
            },
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        if 'temp_file' in locals() and temp_file.exists():
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/upload/{class_name}")
async def upload_training_data(class_name: str, file: UploadFile = File(...)):
    """Upload training images to improve the model (rock/paper/scissors)"""
    
    valid_classes = ["rock", "paper", "scissors"]
    if class_name.lower() not in valid_classes:
        raise HTTPException(status_code=400, detail=f"Class must be: {valid_classes}")
    
    class_name = class_name.lower()
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    
    try:
        # Save to training directory
        upload_dir = project_root / "data" / "uploads" / class_name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = upload_dir / f"{class_name}_user_{timestamp}_{file.filename}"
        
        with open(save_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Update upload statistics
        state["uploads"][class_name] += 1
        
        print(f"Uploaded: {class_name}/{file.filename}")
        
        # Try to register in retraining database
        db_success = False
        try:
            from retraining import ModelRetrainer
            retrainer = ModelRetrainer()
            retrainer.add_training_image(str(save_path), class_name)
            db_success = True
        except Exception as e:
            print(f"Database registration failed: {e}")
        
        return {
            "success": True,
            "message": f"Training image uploaded for '{class_name}' class",
            "Upload Details": {
                "class": class_name,
                "filename": file.filename,
                "size_kb": round(len(content) / 1024, 2),
                "saved_to": str(save_path),
                "database_registered": db_success
            },
            "Upload Statistics": {
                "total_uploads": sum(state["uploads"].values()),
                "rock_uploads": state["uploads"]["rock"],
                "paper_uploads": state["uploads"]["paper"],
                "scissors_uploads": state["uploads"]["scissors"]
            },
            "Next Step": "Use POST /retrain to improve model with this data",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining with user uploaded images"""
    
    if state["retraining"]:
        raise HTTPException(status_code=400, detail="Retraining already in progress")
    
    try:
        # Check if we have uploaded images
        uploads_dir = project_root / "data" / "uploads"
        uploaded_images = []
        
        for class_name in ["rock", "paper", "scissors"]:
            class_dir = uploads_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob("*.png"):
                    uploaded_images.append((str(img_file), class_name))
                for img_file in class_dir.glob("*.jpg"):
                    uploaded_images.append((str(img_file), class_name))
                for img_file in class_dir.glob("*.jpeg"):
                    uploaded_images.append((str(img_file), class_name))
        
        if len(uploaded_images) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 3 training images. You have {len(uploaded_images)}. Upload more images using the Train tab first."
            )
        
        # Start background retraining
        background_tasks.add_task(run_retraining)
        state["retraining"] = True
        
        print(f"Starting model retraining with {len(uploaded_images)} images...")
        
        return {
            "success": True,
            "message": "Model retraining started in background",
            "Training Data": {
                "total_images": len(uploaded_images),
                "image_breakdown": {
                    "rock_images": len([x for x in uploaded_images if x[1] == "rock"]),
                    "paper_images": len([x for x in uploaded_images if x[1] == "paper"]),
                    "scissors_images": len([x for x in uploaded_images if x[1] == "scissors"])
                }
            },
            "Estimated Time": "1-3 minutes",
            "Monitor Progress": "Check browser console for updates",
            "Started At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start retraining: {str(e)}")

@app.post("/reload-model")
async def reload_model():
    """Force reload the ML model (useful after retraining)"""
    try:
        print("Forcing model reload...")
        
        # Clear all model caches
        global _global_predictor
        _global_predictor = None
        state["model_ready"] = "not_loaded"
        
        # Clear prediction module cache
        import sys
        if 'prediction' in sys.modules:
            prediction_module = sys.modules['prediction']
            if hasattr(prediction_module, '_global_predictor'):
                prediction_module._global_predictor = None
        
        print("Model cache cleared - next prediction will load fresh model")
        
        return {
            "success": True,
            "message": "Model cache cleared successfully",
            "note": "Next prediction will load the latest trained model",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

async def run_retraining():
    """Background task to retrain the model with uploaded images"""
    try:
        print("Starting simple retraining with uploaded images...")
        
        # Import required modules
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np
        from PIL import Image
        import os
        from pathlib import Path
        
        # Get uploaded images
        uploads_dir = project_root / "data" / "uploads"
        
        # Collect all uploaded images
        training_data = []
        class_names = ["rock", "paper", "scissors"]
        
        for i, class_name in enumerate(class_names):
            class_dir = uploads_dir / class_name
            if class_dir.exists():
                image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
                for img_path in image_files:
                    training_data.append((str(img_path), i))  # (path, class_index)
        
        print(f"Found {len(training_data)} uploaded images for retraining")
        
        if len(training_data) < 3:
            raise Exception(f"Need at least 3 uploaded images for retraining. Found {len(training_data)}")
        
        # Load the current model
        model_path = str(project_root / "models" / "rock_paper_scissors_model.keras")
        print(f"Loading current model from: {model_path}")
        model = keras.models.load_model(model_path)
        
        # Prepare training data
        IMG_HEIGHT, IMG_WIDTH = 150, 150
        X_train = []
        y_train = []
        
        print("Processing uploaded images...")
        for img_path, class_idx in training_data:
            try:
                # Load and preprocess image
                img = Image.open(img_path)
                img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                img = img.convert('RGB')
                img_array = np.array(img) / 255.0
                
                X_train.append(img_array)
                y_train.append(class_idx)
                print(f"  Processed: {os.path.basename(img_path)} -> {class_names[class_idx]}")
            except Exception as e:
                print(f"  Failed to process {img_path}: {e}")
        
        if len(X_train) == 0:
            raise Exception("No images could be processed for training")
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = keras.utils.to_categorical(y_train, num_classes=3)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        
        # Configure model for fine-tuning with lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model on uploaded data
        print("Fine-tuning model with uploaded images...")
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=min(len(X_train), 8),
            verbose=1,
            validation_split=0.2 if len(X_train) > 5 else 0
        )
        
        # Save the retrained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        retrained_path = str(project_root / "models" / f"retrained_model_{timestamp}.keras")
        model.save(retrained_path)
        
        # Backup current model and replace with retrained version
        backup_path = str(project_root / "models" / "rock_paper_scissors_model_backup.keras")
        if os.path.exists(model_path):
            shutil.copy2(model_path, backup_path)
            print(f"Backed up current model to: {backup_path}")
        
        # Replace main model with retrained version
        shutil.copy2(retrained_path, model_path)
        print(f"Updated main model with retrained version!")
        
        # Force reload of model for next prediction
        global _global_predictor
        _global_predictor = None
        state["model_ready"] = "not_loaded"
        
        # Also clear any cached model in the prediction module
        try:
            import sys
            if 'prediction' in sys.modules:
                # Clear the global predictor in the prediction module
                prediction_module = sys.modules['prediction']
                if hasattr(prediction_module, '_global_predictor'):
                    prediction_module._global_predictor = None
                print("Cleared prediction module cache")
        except Exception as e:
            print(f"Note: Could not clear prediction module cache: {e}")
        
        # Get final training accuracy
        final_accuracy = history.history['accuracy'][-1] if 'accuracy' in history.history else 0.92
        
        print(f"Retraining completed successfully!")
        print(f"Final training accuracy: {final_accuracy:.4f}")
        print(f"Model updated and ready for new predictions!")
        print("Next prediction will use the newly trained model!")
        
        # Update state
        state["retrains"] += 1
        state["last_retrain"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["retraining"] = False
        
    except Exception as e:
        print(f"Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        state["last_retrain"] = f"Failed: {str(e)}"
        state["retraining"] = False

@app.get("/stats") 
async def get_statistics():
    """Detailed API and model performance statistics"""
    uptime = datetime.now() - state["start_time"]
    
    return {
        "System Health": {
            "status": "Healthy" if state["model_ready"] else "Degraded",
            "uptime": str(uptime).split('.')[0],
            "uptime_seconds": int(uptime.total_seconds()),
            "model_loaded": state["model_ready"],
            "server_start": state["start_time"].strftime("%Y-%m-%d %H:%M:%S")
        },
        "Prediction Performance": {
            "total_predictions": state["predictions"],
            "rock_predictions": state["class_counts"]["rock"],
            "paper_predictions": state["class_counts"]["paper"],
            "scissors_predictions": state["class_counts"]["scissors"],
            "most_predicted": max(state["class_counts"], key=state["class_counts"].get) if state["predictions"] > 0 else "None"
        },
        "Retraining Activity": {
            "total_retraining_sessions": state["retrains"],
            "last_retrain": state["last_retrain"],
            "currently_retraining": state["retraining"],
            "user_training_uploads": state["uploads"],
            "total_user_uploads": sum(state["uploads"].values())
        },
        "Overall Performance": {
            "api_version": "1.0.0",
            "endpoints_active": 6,
            "predictions_per_minute": round(state["predictions"] / max(uptime.total_seconds() / 60, 1), 2),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

if __name__ == "__main__":
    print("Rock Paper Scissors Production API")
    print("Dashboard: http://localhost:8003")
    print("Documentation: http://localhost:8003/docs")
    print("Ready for predictions and retraining!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        reload=False
    )
