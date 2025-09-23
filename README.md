# Age Detection System

A state-of-the-art age detection system using multiple deep learning approaches including DeepFace, MediaPipe, and OpenCV DNN models. This project provides both command-line and web interfaces for accurate age estimation from facial images.

## Features

- **Multiple Detection Methods**: DeepFace, MediaPipe, and OpenCV DNN
- **Ensemble Predictions**: Combines multiple models for improved accuracy
- **Web Interfaces**: Streamlit and Gradio UIs for interactive use
- **Batch Processing**: Process multiple images simultaneously
- **Video Processing**: Real-time age detection in video streams
- **Database Storage**: SQLite database for storing results and analytics
- **Performance Analysis**: Comprehensive accuracy metrics and reporting
- **Mock Data Generation**: Synthetic data for testing and development

## Requirements

- Python 3.8+
- OpenCV 4.8+
- PyTorch 2.1+
- DeepFace
- MediaPipe
- Streamlit (for web UI)
- Gradio (for alternative web UI)

## 🛠️ Installation

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/modern-age-detection.git
cd modern-age-detection

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n age-detection python=3.10
conda activate age-detection

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Docker

```bash
# Build Docker image
docker build -t modern-age-detection .

# Run container
docker run -p 7860:7860 modern-age-detection
```

## Quick Start

### Command Line Usage

```python
from modern_age_detector import ModernAgeDetector

# Initialize detector
detector = ModernAgeDetector()

# Process single image
results = detector.process_image("path/to/image.jpg")
print(f"Detected {results['faces_detected']} faces")

for prediction in results['predictions']:
    print(f"Age: {prediction['age']} ({prediction['age_range']})")
```

### Web Interface (Streamlit)

```bash
# Run Streamlit app
streamlit run streamlit_app.py

# Or use the command
age-detection-streamlit
```

### Web Interface (Gradio)

```bash
# Run Gradio app
python gradio_app.py

# Or use the command
age-detection-gradio
```

## Usage Examples

### Single Image Processing

```python
import cv2
from modern_age_detector import ModernAgeDetector

detector = ModernAgeDetector()

# Load and process image
image = cv2.imread("sample.jpg")
results = detector.process_image("sample.jpg")

# Display results
detector.visualize_results("sample.jpg", results)
```

### Batch Processing

```python
from advanced_features import BatchProcessor

batch_processor = BatchProcessor(detector)

# Process entire directory
results = batch_processor.process_directory("images/", max_images=100)

print(f"Processed {results['successful_images']} images")
print(f"Total faces detected: {results['summary']['total_faces_detected']}")
```

### Video Processing

```python
from advanced_features import VideoProcessor

video_processor = VideoProcessor(detector)

# Process video file
results = video_processor.process_video(
    "input_video.mp4", 
    "output_video.mp4",
    frame_skip=5
)

print(f"Processed {results['processed_frames']} frames")
```

### Performance Analysis

```python
from advanced_features import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(detector)

# Load test data
test_data = [
    {"image_path": "test1.jpg", "true_age": 25, "age_range": "20-29"},
    {"image_path": "test2.jpg", "true_age": 45, "age_range": "40-49"},
]

# Calculate accuracy
metrics = analyzer.calculate_accuracy(test_data)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

## Web Interface Features

### Streamlit App
- **Single Image**: Upload and process individual images
- **Batch Processing**: Upload multiple images for bulk processing
- **Analytics Dashboard**: View statistics and performance metrics
- **Database Management**: Manage stored results and sample data
- **Settings**: Configure detection parameters

### Gradio App
- **Interactive Interface**: Simple drag-and-drop interface
- **Real-time Results**: Instant age detection with visual feedback
- **Batch Upload**: Process multiple images simultaneously
- **Statistics View**: System performance and usage statistics

## Performance Metrics

The system provides comprehensive performance analysis:

- **Overall Accuracy**: Percentage of correct age predictions
- **Age Range Accuracy**: Accuracy of age range classification
- **Mean Age Error**: Average error in years
- **Confidence Scores**: Reliability metrics for predictions
- **Processing Speed**: Frames per second for video processing

## Database Schema

The system uses SQLite for data storage:

### Detections Table
- `id`: Primary key
- `image_path`: Path to processed image
- `detected_age`: Predicted age
- `age_range`: Age range category
- `confidence`: Confidence score
- `method`: Detection method used
- `timestamp`: Processing timestamp
- `face_count`: Number of faces detected

### Sample Images Table
- `id`: Primary key
- `image_path`: Path to sample image
- `true_age`: Ground truth age
- `age_range`: True age range
- `source`: Data source
- `added_date`: Date added

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=modern_age_detector

# Run specific test file
pytest tests/test_detector.py
```

### Generate Mock Data

```bash
# Create sample database
python mock_database.py

# This creates:
# - sample_images/ directory with synthetic faces
# - age_detection.db SQLite database
# - sample_video.mp4 test video
```

## Advanced Features

### Ensemble Methods
The system combines multiple detection approaches:
- DeepFace for high accuracy
- MediaPipe for fast detection
- Heuristic methods as fallback
- Weighted averaging based on confidence

### Video Processing
- Real-time face detection
- Frame skipping for efficiency
- Output video with annotations
- Comprehensive statistics

### Batch Processing
- Directory processing
- File list processing
- Progress tracking
- Error handling and reporting

## 🔧 Configuration

### Environment Variables

```bash
# Set device preference
export CUDA_VISIBLE_DEVICES=0

# Set database path
export AGE_DETECTION_DB_PATH=/path/to/database.db

# Set log level
export LOG_LEVEL=INFO
```

### Configuration File

Create `config.yaml`:

```yaml
detection:
  confidence_threshold: 0.5
  preferred_method: "ensemble"
  max_faces: 10

video:
  frame_skip: 5
  max_frames: 1000
  output_format: "mp4"

database:
  path: "age_detection.db"
  backup_interval: 24  # hours
```

## Deployment

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py --server.port 8501

# Run Gradio app
python gradio_app.py
```

### Docker Deployment

```bash
# Build image
docker build -t modern-age-detection .

# Run container
docker run -p 8501:8501 -p 7860:7860 modern-age-detection
```

### Cloud Deployment

The app can be deployed on:
- **Heroku**: Use the included `Procfile`
- **AWS**: Use EC2 with Docker
- **Google Cloud**: Use Cloud Run
- **Azure**: Use Container Instances

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
black .
flake8 .
mypy modern_age_detector.py

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) for age detection models
- [MediaPipe](https://mediapipe.dev/) for face detection
- [OpenCV](https://opencv.org/) for computer vision utilities
- [Streamlit](https://streamlit.io/) for web interface
- [Gradio](https://gradio.app/) for alternative web interface
 
 
## Roadmap

- [ ] Real-time webcam processing
- [ ] Mobile app integration
- [ ] Advanced age estimation models
- [ ] Multi-language support
- [ ] Cloud API service
- [ ] Edge device optimization


# Age-Detection-System
