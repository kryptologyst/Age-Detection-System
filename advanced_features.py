import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
import threading
import queue
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Process video files for age detection"""
    
    def __init__(self, detector):
        self.detector = detector
        self.frame_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        self.processing = False
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     frame_skip: int = 5, max_frames: Optional[int] = None) -> Dict:
        """Process a video file for age detection"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {fps} FPS, {total_frames} frames, {width}x{height}")
        
        # Setup output video if specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        detection_results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if specified
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Limit frames if specified
                if max_frames and processed_frames >= max_frames:
                    break
                
                # Process frame
                results = self._process_frame(frame, frame_count)
                detection_results.append({
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'results': results
                })
                
                # Draw results on frame
                annotated_frame = self._draw_results(frame, results)
                
                # Write to output video
                if out:
                    out.write(annotated_frame)
                
                processed_frames += 1
                frame_count += 1
                
                # Progress logging
                if processed_frames % 30 == 0:  # Every 30 frames
                    progress = (processed_frames / min(total_frames, max_frames or total_frames)) * 100
                    logger.info(f"Processed {processed_frames} frames ({progress:.1f}%)")
        
        finally:
            cap.release()
            if out:
                out.release()
        
        # Calculate summary statistics
        summary = self._calculate_video_summary(detection_results)
        
        return {
            'video_path': video_path,
            'output_path': output_path,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'fps': fps,
            'duration': total_frames / fps,
            'detection_results': detection_results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """Process a single frame"""
        try:
            # Save temporary frame
            temp_path = f"temp_frame_{frame_number}.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Process with detector
            results = self.detector.process_image(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            return results
        
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            return {'error': str(e), 'faces_detected': 0, 'predictions': []}
    
    def _draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results on frame"""
        annotated_frame = frame.copy()
        
        if results.get('error'):
            cv2.putText(annotated_frame, f"Error: {results['error']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated_frame
        
        for prediction in results.get('predictions', []):
            if prediction['age'] is not None:
                x1, y1, x2, y2 = prediction['face_box']
                
                # Draw rectangle
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add text
                text = f"Age: {prediction['age']} ({prediction['age_range']})"
                confidence_text = f"Conf: {prediction['confidence']:.2f}"
                
                # Draw text background
                cv2.rectangle(annotated_frame, (x1, y1-40), (x2, y1), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(annotated_frame, text, (x1+5, y1-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(annotated_frame, confidence_text, (x1+5, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_frame
    
    def _calculate_video_summary(self, detection_results: List[Dict]) -> Dict:
        """Calculate summary statistics for video processing"""
        if not detection_results:
            return {}
        
        total_faces = sum(r['results'].get('faces_detected', 0) for r in detection_results)
        frames_with_faces = sum(1 for r in detection_results if r['results'].get('faces_detected', 0) > 0)
        
        # Calculate age statistics
        all_ages = []
        all_confidences = []
        
        for result in detection_results:
            for prediction in result['results'].get('predictions', []):
                if prediction.get('age') is not None:
                    all_ages.append(prediction['age'])
                    all_confidences.append(prediction.get('confidence', 0))
        
        summary = {
            'total_faces_detected': total_faces,
            'frames_with_faces': frames_with_faces,
            'face_detection_rate': frames_with_faces / len(detection_results) if detection_results else 0,
            'average_faces_per_frame': total_faces / len(detection_results) if detection_results else 0
        }
        
        if all_ages:
            summary.update({
                'average_age': np.mean(all_ages),
                'age_std': np.std(all_ages),
                'min_age': min(all_ages),
                'max_age': max(all_ages),
                'average_confidence': np.mean(all_confidences)
            })
        
        return summary

class BatchProcessor:
    """Process multiple images in batch"""
    
    def __init__(self, detector):
        self.detector = detector
        self.results = []
        self.errors = []
    
    def process_directory(self, directory_path: str, 
                         file_extensions: List[str] = None,
                         max_images: Optional[int] = None) -> Dict:
        """Process all images in a directory"""
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Get all image files
        image_files = []
        for ext in file_extensions:
            pattern = f"**/*{ext}"
            image_files.extend(Path(directory_path).glob(pattern))
            pattern = f"**/*{ext.upper()}"
            image_files.extend(Path(directory_path).glob(pattern))
        
        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images
        results = []
        errors = []
        
        for i, image_path in enumerate(image_files):
            try:
                logger.info(f"Processing {i+1}/{len(image_files)}: {image_path}")
                
                result = self.detector.process_image(str(image_path))
                result['file_path'] = str(image_path)
                result['file_name'] = image_path.name
                results.append(result)
                
            except Exception as e:
                error = {
                    'file_path': str(image_path),
                    'file_name': image_path.name,
                    'error': str(e)
                }
                errors.append(error)
                logger.error(f"Error processing {image_path}: {e}")
        
        # Calculate summary statistics
        summary = self._calculate_batch_summary(results)
        
        return {
            'directory_path': directory_path,
            'total_images': len(image_files),
            'successful_images': len(results),
            'failed_images': len(errors),
            'results': results,
            'errors': errors,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def process_file_list(self, file_paths: List[str]) -> Dict:
        """Process a list of image files"""
        logger.info(f"Processing {len(file_paths)} images")
        
        results = []
        errors = []
        
        for i, file_path in enumerate(file_paths):
            try:
                logger.info(f"Processing {i+1}/{len(file_paths)}: {file_path}")
                
                result = self.detector.process_image(file_path)
                result['file_path'] = file_path
                result['file_name'] = os.path.basename(file_path)
                results.append(result)
                
            except Exception as e:
                error = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'error': str(e)
                }
                errors.append(error)
                logger.error(f"Error processing {file_path}: {e}")
        
        # Calculate summary statistics
        summary = self._calculate_batch_summary(results)
        
        return {
            'total_images': len(file_paths),
            'successful_images': len(results),
            'failed_images': len(errors),
            'results': results,
            'errors': errors,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_batch_summary(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics for batch processing"""
        if not results:
            return {}
        
        total_faces = sum(r.get('faces_detected', 0) for r in results)
        images_with_faces = sum(1 for r in results if r.get('faces_detected', 0) > 0)
        
        # Calculate age statistics
        all_ages = []
        all_confidences = []
        age_ranges = {}
        
        for result in results:
            for prediction in result.get('predictions', []):
                if prediction.get('age') is not None:
                    all_ages.append(prediction['age'])
                    all_confidences.append(prediction.get('confidence', 0))
                    
                    age_range = prediction.get('age_range', 'Unknown')
                    age_ranges[age_range] = age_ranges.get(age_range, 0) + 1
        
        summary = {
            'total_faces_detected': total_faces,
            'images_with_faces': images_with_faces,
            'face_detection_rate': images_with_faces / len(results) if results else 0,
            'average_faces_per_image': total_faces / len(results) if results else 0,
            'age_range_distribution': age_ranges
        }
        
        if all_ages:
            summary.update({
                'average_age': np.mean(all_ages),
                'age_std': np.std(all_ages),
                'min_age': min(all_ages),
                'max_age': max(all_ages),
                'average_confidence': np.mean(all_confidences)
            })
        
        return summary

class PerformanceAnalyzer:
    """Analyze performance metrics and accuracy"""
    
    def __init__(self, detector):
        self.detector = detector
    
    def calculate_accuracy(self, test_data: List[Dict]) -> Dict:
        """Calculate accuracy metrics against ground truth"""
        if not test_data:
            return {}
        
        correct_predictions = 0
        total_predictions = 0
        age_range_correct = 0
        age_range_total = 0
        
        age_errors = []
        confidence_scores = []
        
        for test_item in test_data:
            image_path = test_item['image_path']
            true_age = test_item['true_age']
            true_age_range = test_item['age_range']
            
            try:
                # Process image
                results = self.detector.process_image(image_path)
                
                for prediction in results.get('predictions', []):
                    if prediction.get('age') is not None:
                        predicted_age = prediction['age']
                        predicted_range = prediction['age_range']
                        confidence = prediction.get('confidence', 0)
                        
                        # Check age range accuracy
                        age_range_total += 1
                        if predicted_range == true_age_range:
                            age_range_correct += 1
                        
                        # Check age accuracy (within 5 years)
                        total_predictions += 1
                        if abs(predicted_age - true_age) <= 5:
                            correct_predictions += 1
                        
                        # Store error for analysis
                        age_errors.append(abs(predicted_age - true_age))
                        confidence_scores.append(confidence)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
        
        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        range_accuracy = age_range_correct / age_range_total if age_range_total > 0 else 0
        mean_error = np.mean(age_errors) if age_errors else 0
        mean_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'age_range_accuracy': range_accuracy,
            'mean_age_error': mean_error,
            'mean_confidence': mean_confidence,
            'age_errors': age_errors,
            'confidence_scores': confidence_scores
        }
    
    def generate_performance_report(self, test_data: List[Dict]) -> str:
        """Generate a comprehensive performance report"""
        metrics = self.calculate_accuracy(test_data)
        
        if not metrics:
            return "No test data available for performance analysis."
        
        report = f"""
# Performance Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Metrics
- Total Predictions: {metrics['total_predictions']}
- Correct Predictions: {metrics['correct_predictions']}
- Overall Accuracy: {metrics['accuracy']:.2%}
- Age Range Accuracy: {metrics['age_range_accuracy']:.2%}
- Mean Age Error: {metrics['mean_age_error']:.1f} years
- Mean Confidence: {metrics['mean_confidence']:.3f}

## Detailed Analysis
- Age errors range: {min(metrics['age_errors']) if metrics['age_errors'] else 'N/A'} to {max(metrics['age_errors']) if metrics['age_errors'] else 'N/A'} years
- Confidence scores range: {min(metrics['confidence_scores']) if metrics['confidence_scores'] else 'N/A'} to {max(metrics['confidence_scores']) if metrics['confidence_scores'] else 'N/A'}

## Recommendations
"""
        
        if metrics['accuracy'] < 0.7:
            report += "- Consider improving model training or using ensemble methods\n"
        
        if metrics['mean_confidence'] < 0.5:
            report += "- Low confidence scores suggest need for better preprocessing\n"
        
        if metrics['mean_age_error'] > 10:
            report += "- High age error suggests need for more accurate models\n"
        
        return report

def main():
    """Main function to demonstrate advanced features"""
    from modern_age_detector import ModernAgeDetector
    
    print("Advanced Age Detection Features Demo")
    print("=" * 40)
    
    # Initialize detector
    detector = ModernAgeDetector()
    
    # Create sample data
    from mock_database import MockDatabaseGenerator
    generator = MockDatabaseGenerator()
    generator.create_mock_database()
    
    # Test video processing
    print("\n🎬 Testing video processing...")
    video_processor = VideoProcessor(detector)
    
    if os.path.exists("sample_video.mp4"):
        try:
            video_results = video_processor.process_video(
                "sample_video.mp4", 
                "output_video.mp4",
                frame_skip=10,
                max_frames=50
            )
            print(f"✅ Video processed: {video_results['processed_frames']} frames")
            print(f"📊 Summary: {video_results['summary']}")
        except Exception as e:
            print(f"❌ Video processing error: {e}")
    
    # Test batch processing
    print("\n📁 Testing batch processing...")
    batch_processor = BatchProcessor(detector)
    
    if os.path.exists("sample_images"):
        try:
            batch_results = batch_processor.process_directory(
                "sample_images",
                max_images=10
            )
            print(f"✅ Batch processed: {batch_results['successful_images']} images")
            print(f"📊 Summary: {batch_results['summary']}")
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
    
    # Test performance analysis
    print("\n📊 Testing performance analysis...")
    analyzer = PerformanceAnalyzer(detector)
    
    # Get test data from database
    conn = sqlite3.connect("age_detection.db")
    cursor = conn.cursor()
    cursor.execute('SELECT image_path, true_age, age_range FROM sample_images LIMIT 5')
    test_data = [{'image_path': row[0], 'true_age': row[1], 'age_range': row[2]} 
                 for row in cursor.fetchall()]
    conn.close()
    
    if test_data:
        try:
            performance_report = analyzer.generate_performance_report(test_data)
            print("✅ Performance analysis completed")
            print(performance_report)
        except Exception as e:
            print(f"❌ Performance analysis error: {e}")
    
    print("\n🎉 Advanced features demo completed!")

if __name__ == "__main__":
    main()
