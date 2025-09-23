#!/usr/bin/env python3
"""
Modern Age Detection System - Demo Script
Demonstrates all features of the modernized age detection system
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main demonstration function"""
    print("🚀 Modern Age Detection System - Demo")
    print("=" * 50)
    
    try:
        # Import modules
        from modern_age_detector import ModernAgeDetector
        from advanced_features import VideoProcessor, BatchProcessor, PerformanceAnalyzer
        from mock_database import MockDatabaseGenerator
        
        print("✅ All modules imported successfully")
        
        # Initialize detector
        print("\n🔧 Initializing age detector...")
        detector = ModernAgeDetector()
        print("✅ Age detector initialized")
        
        # Create sample database
        print("\n📊 Creating sample database...")
        generator = MockDatabaseGenerator()
        generator.create_mock_database()
        print("✅ Sample database created")
        
        # Get statistics
        print("\n📈 System Statistics:")
        stats = detector.get_statistics()
        print(f"   Total detections: {stats['total_detections']}")
        print(f"   Average confidence: {stats['average_confidence']:.3f}")
        print(f"   Methods used: {list(stats['method_distribution'].keys())}")
        
        # Test batch processing
        print("\n📁 Testing batch processing...")
        if os.path.exists("sample_images"):
            batch_processor = BatchProcessor(detector)
            batch_results = batch_processor.process_directory("sample_images", max_images=5)
            print(f"✅ Processed {batch_results['successful_images']} images")
            print(f"   Total faces detected: {batch_results['summary']['total_faces_detected']}")
        else:
            print("⚠️  Sample images directory not found")
        
        # Test video processing
        print("\n🎬 Testing video processing...")
        if os.path.exists("sample_video.mp4"):
            video_processor = VideoProcessor(detector)
            video_results = video_processor.process_video(
                "sample_video.mp4", 
                "output_demo.mp4",
                frame_skip=10,
                max_frames=20
            )
            print(f"✅ Processed {video_results['processed_frames']} frames")
            print(f"   Face detection rate: {video_results['summary']['face_detection_rate']:.2%}")
        else:
            print("⚠️  Sample video not found")
        
        # Test performance analysis
        print("\n📊 Testing performance analysis...")
        analyzer = PerformanceAnalyzer(detector)
        
        # Get test data from database
        import sqlite3
        conn = sqlite3.connect("age_detection.db")
        cursor = conn.cursor()
        cursor.execute('SELECT image_path, true_age, age_range FROM sample_images LIMIT 3')
        test_data = [{'image_path': row[0], 'true_age': row[1], 'age_range': row[2]} 
                     for row in cursor.fetchall()]
        conn.close()
        
        if test_data:
            performance_report = analyzer.generate_performance_report(test_data)
            print("✅ Performance analysis completed")
            print("\n📋 Performance Report:")
            print(performance_report)
        else:
            print("⚠️  No test data available")
        
        # Web interface information
        print("\n🌐 Web Interfaces Available:")
        print("   Streamlit: streamlit run streamlit_app.py")
        print("   Gradio: python gradio_app.py")
        
        # Docker information
        print("\n🐳 Docker Deployment:")
        print("   Build: docker build -t modern-age-detection .")
        print("   Run Streamlit: docker run -p 8501:8501 modern-age-detection streamlit")
        print("   Run Gradio: docker run -p 7860:7860 modern-age-detection gradio")
        
        print("\n🎉 Demo completed successfully!")
        print("\n📚 Next Steps:")
        print("   1. Run 'streamlit run streamlit_app.py' for web interface")
        print("   2. Run 'python gradio_app.py' for alternative interface")
        print("   3. Check README.md for detailed documentation")
        print("   4. Run 'pytest' to execute test suite")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)
        sys.exit(1)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'cv2', 'numpy', 'torch', 'PIL', 'streamlit', 'gradio',
        'matplotlib', 'seaborn', 'pandas', 'sqlite3'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'sqlite3':
                import sqlite3
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    return True

if __name__ == "__main__":
    print("🔍 Checking dependencies...")
    if check_dependencies():
        print("✅ All dependencies available")
        main()
    else:
        sys.exit(1)
