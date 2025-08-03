// Configuration
const API_BASE_URL = 'http://localhost:8003';

// Global variables
let selectedClass = 'rock';
let uploadQueue = [];
let isRetraining = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    checkApiStatus();
    loadStatistics();
}

// Event Listeners Setup
function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const section = e.target.getAttribute('onclick').match(/'([^']+)'/)[1];
            showSection(section);
        });
    });

    // Prediction upload
    const uploadArea = document.getElementById('upload-area');
    const predictFile = document.getElementById('predict-file');

    uploadArea.addEventListener('click', () => predictFile.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    predictFile.addEventListener('change', handlePredictFileSelect);

    // Training upload
    const trainUploadArea = document.getElementById('train-upload-area');
    const trainFiles = document.getElementById('train-files');

    trainUploadArea.addEventListener('click', () => trainFiles.click());
    trainUploadArea.addEventListener('dragover', handleTrainDragOver);
    trainUploadArea.addEventListener('dragleave', handleTrainDragLeave);
    trainUploadArea.addEventListener('drop', handleTrainDrop);
    trainFiles.addEventListener('change', handleTrainFileSelect);

    // Class selection
    document.querySelectorAll('.class-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.class-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            selectedClass = this.getAttribute('data-class');
            document.getElementById('selected-class').textContent = 
                selectedClass.charAt(0).toUpperCase() + selectedClass.slice(1);
        });
    });
}

// Navigation
function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });

    // Show selected section
    document.getElementById(sectionName + '-section').classList.add('active');

    // Update navigation buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    event.target.classList.add('active');

    // Load section-specific data
    if (sectionName === 'stats') {
        loadStatistics();
    }
}

// API Status Check
async function checkApiStatus() {
    const statusIndicator = document.querySelector('.status-indicator');
    const statusText = document.querySelector('.status-text');

    try {
        const response = await fetch(`${API_BASE_URL}/`);
        if (response.ok) {
            statusIndicator.classList.add('online');
            statusText.textContent = 'API Connected - Ready to use';
        } else {
            throw new Error('API not responding');
        }
    } catch (error) {
        statusIndicator.classList.add('offline');
        statusText.textContent = 'API Disconnected - Please check connection';
        showToast('API connection failed. Please ensure the API server is running.', 'error');
    }
}

// Drag and Drop Handlers for Prediction
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && isImageFile(files[0])) {
        handleImagePrediction(files[0]);
    } else {
        showToast('Please drop a valid image file (JPG, PNG)', 'error');
    }
}

function handlePredictFileSelect(e) {
    const file = e.target.files[0];
    if (file && isImageFile(file)) {
        handleImagePrediction(file);
    }
}

// Image Prediction
async function handleImagePrediction(file) {
    showLoading('Analyzing image...');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/predict/`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        displayPredictionResults(file, result);
        showToast('Prediction completed successfully!', 'success');
    } catch (error) {
        console.error('Prediction error:', error);
        showToast('Failed to predict image. Please try again.', 'error');
    } finally {
        hideLoading();
    }
}

function displayPredictionResults(file, result) {
    const resultsDiv = document.getElementById('prediction-results');
    const previewImg = document.getElementById('preview-image');
    const predictionClass = document.getElementById('prediction-class');
    const confidence = document.getElementById('confidence');
    const predictionIcon = document.getElementById('prediction-icon');

    // Show image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Display prediction
    const predictedClass = result.predicted_class;
    const confidenceValue = (result.confidence * 100).toFixed(1);

    predictionClass.textContent = predictedClass.charAt(0).toUpperCase() + predictedClass.slice(1);
    confidence.textContent = `${confidenceValue}% Confidence`;

    // Update icon
    const iconMap = {
        'rock': 'fas fa-hand-rock',
        'paper': 'fas fa-hand-paper',
        'scissors': 'fas fa-hand-scissors'
    };
    predictionIcon.innerHTML = `<i class="${iconMap[predictedClass]}"></i>`;

    // Update probability bars
    const probabilities = result.all_probabilities || {};
    updateProbabilityBars(probabilities);

    // Show results
    resultsDiv.style.display = 'block';
    document.getElementById('upload-area').style.display = 'none';
}

function updateProbabilityBars(probabilities) {
    const classes = ['rock', 'paper', 'scissors'];
    
    classes.forEach(className => {
        const prob = probabilities[className] || 0;
        const percentage = (prob * 100).toFixed(1);
        
        const fillElement = document.getElementById(`${className}-prob`);
        const valueElement = document.getElementById(`${className}-value`);
        
        if (fillElement && valueElement) {
            fillElement.style.width = `${percentage}%`;
            valueElement.textContent = `${percentage}%`;
        }
    });
}

function resetPredict() {
    document.getElementById('prediction-results').style.display = 'none';
    document.getElementById('upload-area').style.display = 'block';
    document.getElementById('predict-file').value = '';
}

// Training Data Upload Handlers
function handleTrainDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleTrainDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleTrainDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = Array.from(e.dataTransfer.files).filter(isImageFile);
    if (files.length > 0) {
        addFilesToQueue(files);
    } else {
        showToast('Please drop valid image files (JPG, PNG)', 'error');
    }
}

function handleTrainFileSelect(e) {
    const files = Array.from(e.target.files).filter(isImageFile);
    if (files.length > 0) {
        addFilesToQueue(files);
    }
}

function addFilesToQueue(files) {
    files.forEach(file => {
        uploadQueue.push({
            file: file,
            class: selectedClass,
            status: 'pending',
            id: Date.now() + Math.random()
        });
    });

    updateQueueDisplay();
    showToast(`${files.length} file(s) added to upload queue`, 'success');
}

function updateQueueDisplay() {
    const queueDiv = document.getElementById('upload-queue');
    const queueItems = document.getElementById('queue-items');

    if (uploadQueue.length === 0) {
        queueDiv.style.display = 'none';
        return;
    }

    queueDiv.style.display = 'block';
    queueItems.innerHTML = '';

    uploadQueue.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'queue-item';
        
        const reader = new FileReader();
        reader.onload = (e) => {
            itemDiv.innerHTML = `
                <img src="${e.target.result}" alt="Preview">
                <div class="queue-item-info">
                    <div class="queue-item-name">${item.file.name}</div>
                    <div class="queue-item-size">${formatFileSize(item.file.size)} • ${item.class}</div>
                </div>
                <div class="queue-item-status ${item.status}">${item.status}</div>
            `;
        };
        reader.readAsDataURL(item.file);

        queueItems.appendChild(itemDiv);
    });
}

async function uploadAllTrainingImages() {
    if (uploadQueue.length === 0) {
        showToast('No files in queue', 'warning');
        return;
    }

    const uploadBtn = document.getElementById('upload-all-btn');
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';

    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < uploadQueue.length; i++) {
        const item = uploadQueue[i];
        item.status = 'uploading';
        updateQueueDisplay();

        try {
            const formData = new FormData();
            formData.append('file', item.file);

            const response = await fetch(`${API_BASE_URL}/upload/${item.class}/`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                item.status = 'success';
                successCount++;
            } else {
                throw new Error(`Upload failed: ${response.status}`);
            }
        } catch (error) {
            console.error('Upload error:', error);
            item.status = 'error';
            errorCount++;
        }

        updateQueueDisplay();
    }

    // Clear queue and reset
    uploadQueue = [];
    setTimeout(() => {
        updateQueueDisplay();
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Upload All Images';
    }, 2000);

    if (errorCount === 0) {
        showToast(`Successfully uploaded ${successCount} images!`, 'success');
    } else {
        showToast(`Upload completed: ${successCount} success, ${errorCount} failed`, 'warning');
    }
}

// Statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats/`);
        if (response.ok) {
            const stats = await response.json();
            updateStatisticsDisplay(stats);
        } else {
            throw new Error('Failed to load statistics');
        }
    } catch (error) {
        console.error('Statistics error:', error);
        showToast('Failed to load statistics', 'error');
    }
}

function updateStatisticsDisplay(stats) {
    // Update stat cards
    document.getElementById('model-accuracy').textContent = 
        stats.model_accuracy ? `${(stats.model_accuracy * 100).toFixed(1)}%` : 'N/A';
    
    document.getElementById('total-images').textContent = 
        stats.total_training_images || '0';
    
    document.getElementById('last-updated').textContent = 
        stats.last_updated || 'Never';
    
    document.getElementById('predictions-made').textContent = 
        stats.predictions_made || '0';

    // Update class distribution
    const classData = stats.class_distribution || {};
    const total = Object.values(classData).reduce((sum, count) => sum + count, 0);
    
    if (total > 0) {
        Object.keys(classData).forEach(className => {
            const count = classData[className];
            const percentage = (count / total) * 100;
            
            const fillElement = document.getElementById(`${className}-dist`);
            const countElement = document.getElementById(`${className}-count`);
            
            if (fillElement && countElement) {
                fillElement.style.width = `${percentage}%`;
                countElement.textContent = count.toString();
            }
        });
    }
}

// Model Retraining
async function startRetraining() {
    if (isRetraining) {
        showToast('Retraining is already in progress', 'warning');
        return;
    }

    const confirmed = confirm(
        'Are you sure you want to start retraining? This may take several minutes and the API will be temporarily unavailable.'
    );

    if (!confirmed) return;

    isRetraining = true;
    const retrainBtn = document.getElementById('retrain-btn');
    const progressDiv = document.getElementById('retrain-progress');
    const progressFill = document.getElementById('retrain-progress-fill');
    const progressText = document.getElementById('retrain-progress-text');

    // Update UI
    retrainBtn.disabled = true;
    retrainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
    progressDiv.style.display = 'block';

    try {
        const useAugmentation = document.getElementById('use-augmentation').checked;
        const epochs = parseInt(document.getElementById('epochs').value);

        const response = await fetch(`${API_BASE_URL}/retrain/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                use_augmentation: useAugmentation,
                epochs: epochs
            })
        });

        if (!response.ok) {
            throw new Error(`Retraining failed: ${response.status}`);
        }

        // Simulate progress (since we don't have real-time updates)
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 90) progress = 90;
            
            progressFill.style.width = `${progress}%`;
            progressText.textContent = `Training in progress... ${Math.round(progress)}%`;
        }, 2000);

        const result = await response.json();

        // Complete progress
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressText.textContent = 'Retraining completed successfully!';

        showToast('Model retrained successfully!', 'success');
        
        // Reload statistics
        setTimeout(() => {
            loadStatistics();
        }, 1000);

    } catch (error) {
        console.error('Retraining error:', error);
        showToast('Retraining failed. Please try again.', 'error');
    } finally {
        isRetraining = false;
        
        setTimeout(() => {
            retrainBtn.disabled = false;
            retrainBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Start Retraining';
            progressDiv.style.display = 'none';
        }, 3000);
    }
}

// Utility Functions
function isImageFile(file) {
    return file.type.startsWith('image/') && 
           (file.type.includes('jpeg') || file.type.includes('jpg') || file.type.includes('png'));
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    loadingText.textContent = message;
    overlay.style.display = 'flex';
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = 'none';
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const iconMap = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    };

    toast.innerHTML = `
        <i class="toast-icon ${iconMap[type]}"></i>
        <span class="toast-message">${message}</span>
    `;

    container.appendChild(toast);

    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            container.removeChild(toast);
        }
    }, 5000);

    // Remove on click
    toast.addEventListener('click', () => {
        if (toast.parentNode) {
            container.removeChild(toast);
        }
    });
}

// Auto-refresh stats every 30 seconds
setInterval(() => {
    if (document.getElementById('stats-section').classList.contains('active')) {
        loadStatistics();
    }
}, 30000);
