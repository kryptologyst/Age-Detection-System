# Modern Age Detection System Tests

import pytest
import numpy as np
import cv2
import os
import tempfile
from PIL import Image
from modern_age_detector import ModernAgeDetector
from advanced_features import VideoProcessor, BatchProcessor, PerformanceAnalyzer
from mock_database import MockDatabaseGenerator

class TestModernAgeDetector:
    """Test cases for ModernAgeDetector"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return ModernAgeDetector()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        # Create a simple test image
        img = Image.new('RGB', (200, 200), color='white')
        return img
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector is not None
        assert detector.device is not None
        assert detector.age_categories is not None
    
    def test_age_range_conversion(self, detector):
        """Test age to age range conversion"""
        assert detector._get_age_range(5) == "3-9"
        assert detector._get_age_range(25) == "20-29"
        assert detector._get_age_range(75) == "70+"
    
    def test_face_detection(self, detector, sample_image):
        """Test face detection functionality"""
        # Convert PIL to OpenCV format
        img_array = np.array(sample_image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        faces = detector.detect_faces(img_cv)
        assert isinstance(faces, list)
    
    def test_database_initialization(self, detector):
        """Test database initialization"""
        assert os.path.exists(detector.db_path)
        
        # Check if tables exist
        import sqlite3
        conn = sqlite3.connect(detector.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'detections' in tables
        assert 'sample_images' in tables
        
        conn.close()
    
    def test_statistics(self, detector):
        """Test statistics functionality"""
        stats = detector.get_statistics()
        assert isinstance(stats, dict)
        assert 'total_detections' in stats
        assert 'average_confidence' in stats
        assert 'method_distribution' in stats

class TestMockDatabaseGenerator:
    """Test cases for MockDatabaseGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance for testing"""
        return MockDatabaseGenerator()
    
    def test_generator_initialization(self, generator):
        """Test generator initialization"""
        assert generator.output_dir == "sample_images"
        assert generator.age_categories is not None
        assert len(generator.age_categories) > 0
    
    def test_age_parameters(self, generator):
        """Test age parameter generation"""
        params = generator._get_age_parameters("20-29")
        assert 'face_size' in params
        assert 'mouth_width' in params
        assert params['face_size'] > 0
    
    def test_random_age_generation(self, generator):
        """Test random age generation within ranges"""
        age = generator._get_random_age_in_range("20-29")
        assert 20 <= age <= 29
        
        age = generator._get_random_age_in_range("0-2")
        assert 0 <= age <= 2
    
    def test_synthetic_face_creation(self, generator):
        """Test synthetic face creation"""
        face_img = generator._create_synthetic_face("20-29", 0)
        assert isinstance(face_img, Image.Image)
        assert face_img.size == (200, 200)

class TestVideoProcessor:
    """Test cases for VideoProcessor"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return ModernAgeDetector()
    
    @pytest.fixture
    def video_processor(self, detector):
        """Create video processor instance"""
        return VideoProcessor(detector)
    
    def test_video_processor_initialization(self, video_processor):
        """Test video processor initialization"""
        assert video_processor.detector is not None
        assert video_processor.frame_queue is not None
        assert video_processor.result_queue is not None
    
    def test_frame_processing(self, video_processor):
        """Test single frame processing"""
        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.fill(255)
        
        results = video_processor._process_frame(frame, 0)
        assert isinstance(results, dict)
        assert 'faces_detected' in results
        assert 'predictions' in results

class TestBatchProcessor:
    """Test cases for BatchProcessor"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return ModernAgeDetector()
    
    @pytest.fixture
    def batch_processor(self, detector):
        """Create batch processor instance"""
        return BatchProcessor(detector)
    
    def test_batch_processor_initialization(self, batch_processor):
        """Test batch processor initialization"""
        assert batch_processor.detector is not None
        assert isinstance(batch_processor.results, list)
        assert isinstance(batch_processor.errors, list)
    
    def test_batch_summary_calculation(self, batch_processor):
        """Test batch summary calculation"""
        # Create mock results
        mock_results = [
            {
                'faces_detected': 2,
                'predictions': [
                    {'age': 25, 'confidence': 0.8, 'age_range': '20-29'},
                    {'age': 30, 'confidence': 0.7, 'age_range': '30-39'}
                ]
            },
            {
                'faces_detected': 1,
                'predictions': [
                    {'age': 45, 'confidence': 0.9, 'age_range': '40-49'}
                ]
            }
        ]
        
        summary = batch_processor._calculate_batch_summary(mock_results)
        
        assert summary['total_faces_detected'] == 3
        assert summary['images_with_faces'] == 2
        assert abs(summary['average_age'] - 33.33) < 0.01  # (25+30+45)/3
        assert 'age_range_distribution' in summary

class TestPerformanceAnalyzer:
    """Test cases for PerformanceAnalyzer"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return ModernAgeDetector()
    
    @pytest.fixture
    def analyzer(self, detector):
        """Create performance analyzer instance"""
        return PerformanceAnalyzer(detector)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.detector is not None
    
    def test_accuracy_calculation(self, analyzer):
        """Test accuracy calculation"""
        # Create mock test data
        test_data = [
            {'image_path': 'test1.jpg', 'true_age': 25, 'age_range': '20-29'},
            {'image_path': 'test2.jpg', 'true_age': 45, 'age_range': '40-49'},
        ]
        
        # Mock detector results
        def mock_process_image(path):
            if 'test1' in path:
                return {
                    'predictions': [
                        {'age': 27, 'confidence': 0.8, 'age_range': '20-29'}
                    ]
                }
            else:
                return {
                    'predictions': [
                        {'age': 43, 'confidence': 0.9, 'age_range': '40-49'}
                    ]
                }
        
        analyzer.detector.process_image = mock_process_image
        
        metrics = analyzer.calculate_accuracy(test_data)
        
        assert metrics['total_predictions'] == 2
        assert metrics['correct_predictions'] == 2  # Both within 5 years
        assert metrics['accuracy'] == 1.0
        assert metrics['age_range_accuracy'] == 1.0

# Integration Tests

class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing"""
        # Create detector
        detector = ModernAgeDetector()
        
        # Create sample image
        img = Image.new('RGB', (200, 200), color='white')
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp.name)
            
            try:
                # Process image
                results = detector.process_image(tmp.name)
                
                # Verify results structure
                assert isinstance(results, dict)
                assert 'faces_detected' in results
                assert 'predictions' in results
                # Note: timestamp is only added when faces are detected
                
            finally:
                # Clean up
                os.unlink(tmp.name)
    
    def test_database_operations(self):
        """Test database operations"""
        detector = ModernAgeDetector()
        
        # Test statistics
        stats = detector.get_statistics()
        assert isinstance(stats, dict)
        
        # Test sample database creation
        generator = MockDatabaseGenerator()
        generator.create_mock_database()
        
        # Verify database was created
        assert os.path.exists("age_detection.db")
        
        # Test statistics after sample data creation
        stats_after = detector.get_statistics()
        assert stats_after['total_detections'] >= stats['total_detections']

# Utility Functions

def create_test_image_with_face():
    """Create a test image with a simple face"""
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple face
    draw.ellipse([50, 50, 150, 150], fill='peachpuff', outline='black')
    draw.ellipse([70, 80, 85, 95], fill='black')  # Left eye
    draw.ellipse([115, 80, 130, 95], fill='black')  # Right eye
    draw.arc([90, 110, 110, 130], start=0, end=180, fill='black')  # Mouth
    
    return img

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
