"""
Rock Paper Scissors AI - Load Testing with Locust
===================================================

This script simulates realistic user behavior for performance testing:
- Image predictions with various file sizes
- Statistics monitoring
- Training data uploads
- Model retraining requests

Usage:
    locust -f locustfile.py --host=http://localhost:8003
    
For multiple containers test:
    locust -f locustfile.py --host=http://localhost --users=100 --spawn-rate=10
"""

import os
import random
import io
from PIL import Image
import requests
from locust import HttpUser, task, between
from locust.exception import StopUser


class RockPaperScissorsUser(HttpUser):
    wait_time = between(1, 5)  # Realistic user behavior
    
    def on_start(self):
        """Initialize test user with sample images"""
        print("🚀 Starting RPS AI load test user...")
        self.test_images = self.create_test_images()
        
        # Check if API is available
        try:
            response = self.client.get("/")
            if response.status_code != 200:
                print("❌ API not available, stopping user")
                raise StopUser()
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            raise StopUser()
    
    def create_test_images(self):
        """Create sample images for testing"""
        images = {}
        
        # Create small test images (to avoid large payloads)
        for class_name in ['rock', 'paper', 'scissors']:
            # Create a simple colored square as test image
            img = Image.new('RGB', (150, 150), color=random.choice([
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
            ]))
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=85)
            img_bytes.seek(0)
            
            images[class_name] = img_bytes.getvalue()
        
        return images
    
    @task(10)  # Most common action - predictions
    def predict_image(self):
        """Test image prediction endpoint"""
        try:
            # Select random class image
            class_name = random.choice(['rock', 'paper', 'scissors'])
            image_data = self.test_images[class_name]
            
            # Make prediction request
            with self.client.post("/predict", 
                                files={"file": ("test.jpg", image_data, "image/jpeg")},
                                catch_response=True) as response:
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Validate response structure
                    if "Prediction" in result and "class" in result["Prediction"]:
                        response.success()
                        
                        # Log prediction results occasionally
                        if random.random() < 0.1:  # 10% of requests
                            predicted_class = result["Prediction"]["class"]
                            confidence = result["Prediction"]["confidence"]
                            print(f"✅ Prediction: {predicted_class} ({confidence})")
                    else:
                        response.failure("Invalid response format")
                else:
                    response.failure(f"HTTP {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Prediction error: {e}")
    
    @task(5)  # Moderate frequency - statistics viewing
    def get_statistics(self):
        """Test statistics endpoint"""
        try:
            with self.client.get("/stats", catch_response=True) as response:
                if response.status_code == 200:
                    stats = response.json()
                    
                    # Validate response has expected fields
                    if "System Health" in stats and "Prediction Performance" in stats:
                        response.success()
                        
                        # Log stats occasionally
                        if random.random() < 0.05:  # 5% of requests
                            total_predictions = stats.get("Prediction Performance", {}).get("total_predictions", 0)
                            uptime = stats.get("System Health", {}).get("uptime", "Unknown")
                            print(f"📊 Stats: {total_predictions} predictions, uptime: {uptime}")
                    else:
                        response.failure("Invalid stats format")
                else:
                    response.failure(f"HTTP {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Stats error: {e}")
    
    @task(3)  # Lower frequency - dashboard access
    def get_dashboard(self):
        """Test main dashboard endpoint"""
        try:
            with self.client.get("/", catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
    
    @task(2)  # Even lower frequency - training data upload
    def upload_training_data(self):
        """Test training data upload"""
        try:
            class_name = random.choice(['rock', 'paper', 'scissors'])
            image_data = self.test_images[class_name]
            
            with self.client.post(f"/upload/{class_name}",
                                files={"file": ("train.jpg", image_data, "image/jpeg")},
                                catch_response=True) as response:
                
                if response.status_code == 200:
                    result = response.json()
                    if "message" in result and "uploaded successfully" in result["message"]:
                        response.success()
                        
                        if random.random() < 0.1:  # Log 10% of uploads
                            print(f"📁 Uploaded training image for {class_name}")
                    else:
                        response.failure("Upload failed")
                else:
                    response.failure(f"HTTP {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Upload error: {e}")
    
    @task(1)  # Rare - model retraining (heavy operation)
    def trigger_retraining(self):
        """Test model retraining (occasionally)"""
        # Only trigger retraining 1% of the time to avoid overload
        if random.random() < 0.01:
            try:
                with self.client.post("/retrain", catch_response=True) as response:
                    if response.status_code == 200:
                        result = response.json()
                        if "message" in result:
                            response.success()
                            print(f"🔄 Retraining triggered successfully")
                        else:
                            response.failure("Retraining response invalid")
                    else:
                        response.failure(f"HTTP {response.status_code}")
                        
            except Exception as e:
                print(f"❌ Retraining error: {e}")


class HeavyLoadUser(HttpUser):
    """Heavy load user for stress testing"""
    wait_time = between(0.1, 0.5)  # Very aggressive
    
    def on_start(self):
        """Quick setup for heavy load"""
        # Create single test image
        img = Image.new('RGB', (150, 150), color=(128, 128, 128))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=75)
        img_bytes.seek(0)
        self.test_image = img_bytes.getvalue()
    
    @task
    def rapid_predictions(self):
        """Rapid-fire predictions for stress testing"""
        try:
            self.client.post("/predict", 
                           files={"file": ("stress.jpg", self.test_image, "image/jpeg")},
                           timeout=10)
        except Exception:
            pass  # Ignore errors in stress test


# Custom test scenarios
class PeakHourUser(HttpUser):
    """Simulate peak hour usage patterns"""
    wait_time = between(0.5, 2)
    
    def on_start(self):
        self.setup_images()
    
    def setup_images(self):
        """Setup realistic image sizes"""
        self.images = []
        
        # Create various sized images (like real users would upload)
        for size in [(100, 100), (150, 150), (200, 200), (300, 300)]:
            img = Image.new('RGB', size, color=random.choice([
                (255, 100, 100), (100, 255, 100), (100, 100, 255)
            ]))
            
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=random.randint(70, 95))
            img_bytes.seek(0)
            
            self.images.append(img_bytes.getvalue())
    
    @task
    def realistic_prediction(self):
        """Make predictions with realistic image sizes"""
        image_data = random.choice(self.images)
        
        try:
            response = self.client.post("/predict",
                                     files={"file": ("real.jpg", image_data, "image/jpeg")})
            
            if response.status_code == 200:
                # Simulate user viewing results
                self.client.get("/stats")
                
        except Exception as e:
            print(f"Peak hour error: {e}")


if __name__ == "__main__":
    print("""
🧪 Rock Paper Scissors AI - Load Testing Suite
==============================================

Available user classes:
- RockPaperScissorsUser: Realistic user behavior simulation
- HeavyLoadUser: Stress testing with rapid requests  
- PeakHourUser: Peak hour usage patterns

Run with:
    locust -f locustfile.py --host=http://localhost:8003
    
For Docker scaling test:
    locust -f locustfile.py --host=http://localhost --users=500 --spawn-rate=50
    """)
