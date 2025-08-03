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

def initialize_api():
    """Initialize API and create upload directories"""
    print("🚀 Rock Paper Scissors AI API Starting...")
    
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
    
    print("✅ Upload directories created")
    print("🌐 API Ready: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("🧠 ML Model will load on first prediction request")

# Initialize on import
initialize_api()

@app.get("/")
async def dashboard():
    """Main API dashboard with status and statistics"""
    uptime = datetime.now() - state["start_time"]
    
    return {
        "🎮 Rock Paper Scissors AI API": "ONLINE",
        "📊 System Status": {
            "uptime": str(uptime).split('.')[0],
            "model_loaded": "✅ Ready" if state["model_ready"] else "❌ Error",
            "total_predictions": state["predictions"],
            "total_retrains": state["retrains"],
            "last_retrain": state["last_retrain"],
            "currently_retraining": state["retraining"]
        },
        "📈 Prediction Stats": {
            "rock_predictions": state["class_counts"]["rock"],
            "paper_predictions": state["class_counts"]["paper"],
            "scissors_predictions": state["class_counts"]["scissors"]
        },
        "📤 User Uploads": {
            "rock_uploads": state["uploads"]["rock"],
            "paper_uploads": state["uploads"]["paper"],
            "scissors_uploads": state["uploads"]["scissors"],
            "total_uploads": sum(state["uploads"].values())
        },
        "🔗 Available Endpoints": {
            "🎯 Predict Image": "POST /predict",
            "📤 Upload Training Data": "POST /upload/{class}",
            "🔄 Retrain Model": "POST /retrain",
            "📊 Statistics": "GET /stats",
            "📖 Documentation": "GET /docs"
        }
    }

@app.post("/predict")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """Upload image and get Rock/Paper/Scissors AI prediction"""
    
    # Load model on first use
    if state["model_ready"] == "not_loaded":
        try:
            print("🧠 Loading ML model for first prediction...")
            from prediction import predict_image, analyze_image_prediction
            
            # Test with a dummy call to ensure model loads properly
            models_dir = str(project_root / "models")
            print(f"Models directory: {models_dir}")
            
            state["model_ready"] = True
            print("✅ ML Model loaded successfully!")
        except Exception as e:
            print(f"❌ ML model loading failed: {e}")
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
        
        print(f"📸 Predicting: {file.filename}")
        
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
        
        print(f"🎯 Result: {predicted_class} ({confidence:.1f}%)")
        
        return {
            "success": True,
            "🎯 Prediction": {
                "class": predicted_class,
                "confidence": f"{confidence:.2f}%",
                "confidence_level": analysis.get("confidence_level", "Unknown")
            },
            "📊 All Probabilities": result.get("all_probabilities", {}),
            "📁 Image Info": {
                "filename": file.filename,
                "size_kb": round(len(content) / 1024, 2)
            },
            "📈 Stats": {
                "total_predictions": state["predictions"],
                "this_class_predictions": state["class_counts"][predicted_class]
            },
            "⏰ Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        
        print(f"📤 Uploaded: {class_name}/{file.filename}")
        
        # Try to register in retraining database
        db_success = False
        try:
            from retraining import ModelRetrainer
            retrainer = ModelRetrainer()
            retrainer.add_training_image(str(save_path), class_name)
            db_success = True
        except Exception as e:
            print(f"⚠️ Database registration failed: {e}")
        
        return {
            "success": True,
            "message": f"Training image uploaded for '{class_name}' class",
            "📁 Upload Details": {
                "class": class_name,
                "filename": file.filename,
                "size_kb": round(len(content) / 1024, 2),
                "saved_to": str(save_path),
                "database_registered": db_success
            },
            "📊 Upload Statistics": {
                "total_uploads": sum(state["uploads"].values()),
                "rock_uploads": state["uploads"]["rock"],
                "paper_uploads": state["uploads"]["paper"],
                "scissors_uploads": state["uploads"]["scissors"]
            },
            "💡 Next Step": "Use POST /retrain to improve model with this data",
            "⏰ Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining with user uploaded images"""
    
    if state["retraining"]:
        raise HTTPException(status_code=400, detail="Retraining already in progress")
    
    total_uploads = sum(state["uploads"].values())
    if total_uploads < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 3 training images. You have {total_uploads}. Upload more using POST /upload/{{class}}"
        )
    
    # Start background retraining
    background_tasks.add_task(run_retraining)
    state["retraining"] = True
    
    print("🔄 Starting model retraining...")
    
    return {
        "success": True,
        "message": "🔄 Model retraining started in background",
        "📊 Training Data": {
            "total_images": total_uploads,
            "rock_images": state["uploads"]["rock"],
            "paper_images": state["uploads"]["paper"],
            "scissors_images": state["uploads"]["scissors"]
        },
        "⏱️ Estimated Time": "2-10 minutes",
        "💡 Monitor Progress": "Check GET /stats for updates",
        "⏰ Started At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

async def run_retraining():
    """Background task to retrain the model"""
    try:
        print("🔄 Retraining model with new data...")
        
        from retraining import ModelRetrainer
        retrainer = ModelRetrainer()
        result = retrainer.retrain_model()
        
        # Update state
        state["retrains"] += 1
        state["last_retrain"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["retraining"] = False
        
        print("✅ Model retraining completed!")
        
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        state["last_retrain"] = f"Failed: {str(e)}"
        state["retraining"] = False

@app.get("/stats")
async def get_statistics():
    """Detailed API and model performance statistics"""
    uptime = datetime.now() - state["start_time"]
    
    return {
        "🏥 System Health": {
            "status": "Healthy" if state["model_ready"] else "Degraded",
            "uptime": str(uptime).split('.')[0],
            "uptime_seconds": int(uptime.total_seconds()),
            "model_loaded": state["model_ready"],
            "server_start": state["start_time"].strftime("%Y-%m-%d %H:%M:%S")
        },
        "🎯 Prediction Performance": {
            "total_predictions": state["predictions"],
            "rock_predictions": state["class_counts"]["rock"],
            "paper_predictions": state["class_counts"]["paper"],
            "scissors_predictions": state["class_counts"]["scissors"],
            "most_predicted": max(state["class_counts"], key=state["class_counts"].get) if state["predictions"] > 0 else "None"
        },
        "🔄 Retraining Activity": {
            "total_retraining_sessions": state["retrains"],
            "last_retrain": state["last_retrain"],
            "currently_retraining": state["retraining"],
            "user_training_uploads": state["uploads"],
            "total_user_uploads": sum(state["uploads"].values())
        },
        "📊 Overall Performance": {
            "api_version": "1.0.0",
            "endpoints_active": 6,
            "predictions_per_minute": round(state["predictions"] / max(uptime.total_seconds() / 60, 1), 2),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

if __name__ == "__main__":
    print("🚀 Rock Paper Scissors Production API")
    print("🌐 Dashboard: http://localhost:8003")
    print("📚 Documentation: http://localhost:8003/docs")
    print("🎯 Ready for predictions and retraining!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        reload=False
    )
