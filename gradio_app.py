import gradio as gr
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import sqlite3
from datetime import datetime
import os
import tempfile
from modern_age_detector import ModernAgeDetector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global detector instance
detector = None

def load_detector():
    """Load the age detector"""
    global detector
    try:
        detector = ModernAgeDetector()
        return "✅ Age detector loaded successfully!"
    except Exception as e:
        return f"❌ Failed to load age detector: {e}"

def process_single_image(image):
    """Process a single image for age detection"""
    if detector is None:
        return None, "Age detector not loaded. Please initialize first."
    
    try:
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, image_cv)
            
            # Process image
            results = detector.process_image(tmp_file.name)
            
            # Clean up
            os.unlink(tmp_file.name)
        
        if results.get('error'):
            return None, f"Error: {results['error']}"
        
        # Create result image with bounding boxes
        result_image = image_cv.copy()
        
        for prediction in results['predictions']:
            if prediction['age'] is not None:
                x1, y1, x2, y2 = prediction['face_box']
                
                # Draw rectangle
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add text
                text = f"Age: {prediction['age']} ({prediction['age_range']})"
                confidence_text = f"Conf: {prediction['confidence']:.2f}"
                
                # Draw text background
                cv2.rectangle(result_image, (x1, y1-40), (x2, y1), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(result_image, text, (x1+5, y1-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(result_image, confidence_text, (x1+5, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Convert back to PIL
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_image_rgb)
        
        # Create results text
        results_text = f"Faces detected: {results['faces_detected']}\n\n"
        
        for i, prediction in enumerate(results['predictions']):
            results_text += f"Face {i+1}:\n"
            results_text += f"  Age: {prediction['age']}\n"
            results_text += f"  Age Range: {prediction['age_range']}\n"
            results_text += f"  Confidence: {prediction['confidence']:.2f}\n"
            results_text += f"  Method: {prediction['method']}\n\n"
        
        return result_pil, results_text
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None, f"Error processing image: {e}"

def process_batch_images(images):
    """Process multiple images for age detection"""
    if detector is None:
        return "Age detector not loaded. Please initialize first."
    
    if not images:
        return "No images provided."
    
    results_summary = []
    
    for i, image in enumerate(images):
        try:
            # Convert PIL to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                cv2.imwrite(tmp_file.name, image_cv)
                
                # Process image
                results = detector.process_image(tmp_file.name)
                
                # Clean up
                os.unlink(tmp_file.name)
            
            # Store summary
            results_summary.append({
                'Image': f"Image {i+1}",
                'Faces Detected': results['faces_detected'],
                'Average Age': np.mean([p['age'] for p in results['predictions'] if p['age'] is not None]) if results['predictions'] else None,
                'Average Confidence': np.mean([p['confidence'] for p in results['predictions']]) if results['predictions'] else 0,
                'Error': results.get('error', 'None')
            })
        
        except Exception as e:
            results_summary.append({
                'Image': f"Image {i+1}",
                'Faces Detected': 0,
                'Average Age': None,
                'Average Confidence': 0,
                'Error': str(e)
            })
    
    # Create summary text
    summary_text = "Batch Processing Results:\n\n"
    
    total_images = len(results_summary)
    successful_images = len([r for r in results_summary if r['Faces Detected'] > 0])
    total_faces = sum([r['Faces Detected'] for r in results_summary])
    
    summary_text += f"Total Images: {total_images}\n"
    summary_text += f"Successful: {successful_images}\n"
    summary_text += f"Total Faces: {total_faces}\n\n"
    
    summary_text += "Detailed Results:\n"
    for result in results_summary:
        summary_text += f"{result['Image']}: {result['Faces Detected']} faces"
        if result['Average Age']:
            summary_text += f", Avg Age: {result['Average Age']:.1f}"
        summary_text += f", Avg Conf: {result['Average Confidence']:.2f}\n"
    
    return summary_text

def get_statistics():
    """Get system statistics"""
    if detector is None:
        return "Age detector not loaded."
    
    try:
        stats = detector.get_statistics()
        
        stats_text = "System Statistics:\n\n"
        stats_text += f"Total Detections: {stats.get('total_detections', 0)}\n"
        stats_text += f"Average Confidence: {stats.get('average_confidence', 0):.3f}\n"
        stats_text += f"Detection Methods Used: {len(stats.get('method_distribution', {}))}\n\n"
        
        if stats.get('method_distribution'):
            stats_text += "Method Distribution:\n"
            for method, count in stats.get('method_distribution', {}).items():
                stats_text += f"  {method}: {count} detections\n"
        
        return stats_text
        
    except Exception as e:
        return f"Error getting statistics: {e}"

def create_sample_data():
    """Create sample data in the database"""
    if detector is None:
        return "Age detector not loaded."
    
    try:
        detector.create_sample_database()
        return "✅ Sample data created successfully!"
    except Exception as e:
        return f"❌ Error creating sample data: {e}"

def clear_database():
    """Clear all data from the database"""
    if detector is None:
        return "Age detector not loaded."
    
    try:
        conn = sqlite3.connect(detector.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM detections')
        conn.commit()
        conn.close()
        return "✅ Database cleared successfully!"
    except Exception as e:
        return f"❌ Error clearing database: {e}"

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Modern Age Detection System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 👶 Modern Age Detection System")
        gr.Markdown("Upload images to detect faces and estimate age using advanced deep learning models.")
        
        # Initialize detector
        with gr.Row():
            init_btn = gr.Button("🚀 Initialize Age Detector", variant="primary")
            init_status = gr.Textbox(label="Initialization Status", interactive=False)
        
        init_btn.click(load_detector, outputs=init_status)
        
        # Single image processing
        with gr.Tab("📸 Single Image"):
            gr.Markdown("### Upload a single image for age detection")
            
            with gr.Row():
                with gr.Column():
                    single_image_input = gr.Image(label="Upload Image", type="pil")
                    single_process_btn = gr.Button("🔍 Detect Age", variant="primary")
                
                with gr.Column():
                    single_image_output = gr.Image(label="Detection Results")
                    single_results_text = gr.Textbox(label="Results", lines=10, interactive=False)
            
            single_process_btn.click(
                process_single_image,
                inputs=single_image_input,
                outputs=[single_image_output, single_results_text]
            )
        
        # Batch processing
        with gr.Tab("📁 Batch Processing"):
            gr.Markdown("### Upload multiple images for batch processing")
            
            with gr.Row():
                with gr.Column():
                    batch_images_input = gr.File(
                        label="Upload Multiple Images",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    batch_process_btn = gr.Button("🚀 Process All Images", variant="primary")
                
                with gr.Column():
                    batch_results_text = gr.Textbox(label="Batch Results", lines=15, interactive=False)
            
            batch_process_btn.click(
                process_batch_images,
                inputs=batch_images_input,
                outputs=batch_results_text
            )
        
        # Analytics
        with gr.Tab("📊 Analytics"):
            gr.Markdown("### System Statistics and Analytics")
            
            with gr.Row():
                stats_btn = gr.Button("📈 Get Statistics", variant="primary")
                stats_output = gr.Textbox(label="Statistics", lines=10, interactive=False)
            
            stats_btn.click(get_statistics, outputs=stats_output)
        
        # Database management
        with gr.Tab("🗄️ Database"):
            gr.Markdown("### Database Management")
            
            with gr.Row():
                create_sample_btn = gr.Button("📊 Create Sample Data")
                clear_db_btn = gr.Button("🗑️ Clear Database", variant="stop")
            
            with gr.Row():
                db_status = gr.Textbox(label="Database Status", interactive=False)
            
            create_sample_btn.click(create_sample_data, outputs=db_status)
            clear_db_btn.click(clear_database, outputs=db_status)
        
        # Examples
        with gr.Tab("💡 Examples"):
            gr.Markdown("### Example Images")
            gr.Markdown("""
            **Tips for best results:**
            - Use clear, well-lit images
            - Ensure faces are clearly visible
            - Avoid extreme angles or occlusions
            - Higher resolution images generally work better
            
            **Supported formats:** JPG, PNG, BMP
            """)
        
        # Footer
        gr.Markdown("""
        ---
        **Modern Age Detection System** - Powered by DeepFace, MediaPipe, and OpenCV
        """)
    
    return demo

def main():
    """Main function to run the Gradio app"""
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
