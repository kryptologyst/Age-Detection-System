# Modern Age Detection System

## Project Structure

```
modern-age-detection/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD pipeline
├── modern_age_detector.py   # Core detection system
├── streamlit_app.py        # Streamlit web interface
├── gradio_app.py           # Gradio web interface
├── advanced_features.py    # Video/batch processing
├── mock_database.py       # Mock data generation
├── 0113.py                # Original implementation
├── tests/                 # Test files
├── sample_images/         # Generated sample images
├── age_detection.db       # SQLite database
└── sample_video.mp4       # Test video
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Web Interface**
   ```bash
   # Streamlit
   streamlit run streamlit_app.py
   
   # Gradio
   python gradio_app.py
   ```

3. **Generate Sample Data**
   ```bash
   python mock_database.py
   ```

4. **Test Advanced Features**
   ```bash
   python advanced_features.py
   ```

## API Reference

### ModernAgeDetector

Main class for age detection.

```python
detector = ModernAgeDetector()
results = detector.process_image("image.jpg")
```

**Methods:**
- `process_image(image_path)`: Process single image
- `detect_faces(image)`: Detect faces in image
- `predict_age_ensemble(image, face_box)`: Predict age using ensemble
- `visualize_results(image_path, results)`: Display results
- `get_statistics()`: Get system statistics

### VideoProcessor

Process video files for age detection.

```python
processor = VideoProcessor(detector)
results = processor.process_video("input.mp4", "output.mp4")
```

### BatchProcessor

Process multiple images in batch.

```python
processor = BatchProcessor(detector)
results = processor.process_directory("images/")
```

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `AGE_DETECTION_DB_PATH`: Database file path
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Detection Parameters

- `confidence_threshold`: Minimum confidence for predictions
- `preferred_method`: Primary detection method
- `max_faces`: Maximum faces to detect per image

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use CPU instead of GPU
   - Process images individually

2. **Model Download Issues**
   - Check internet connection
   - Clear model cache
   - Use offline models

3. **Face Detection Fails**
   - Ensure good lighting
   - Check image quality
   - Try different detection methods

### Performance Optimization

1. **Speed Optimization**
   - Use MediaPipe for fast detection
   - Reduce image resolution
   - Skip frames in video processing

2. **Accuracy Improvement**
   - Use ensemble methods
   - Improve image preprocessing
   - Fine-tune confidence thresholds

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## License

MIT License - see LICENSE file for details.
