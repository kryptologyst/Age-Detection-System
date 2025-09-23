import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# Page configuration
st.set_page_config(
    page_title="Modern Age Detection System",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    """Load the age detector with caching"""
    try:
        detector = ModernAgeDetector()
        return detector
    except Exception as e:
        st.error(f"Failed to load age detector: {e}")
        return None

def init_session_state():
    """Initialize session state variables"""
    if 'detector' not in st.session_state:
        st.session_state.detector = load_detector()
    
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = []
    
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {}

def get_database_stats():
    """Get statistics from the database"""
    if st.session_state.detector is None:
        return {}
    
    try:
        return st.session_state.detector.get_statistics()
    except Exception as e:
        st.error(f"Error getting statistics: {e}")
        return {}

def display_image_with_results(image, results, key_prefix=""):
    """Display image with age detection results"""
    if not results or not results.get('predictions'):
        st.warning("No faces detected in the image")
        return
    
    # Create a copy of the image for drawing
    display_image = image.copy()
    
    # Draw bounding boxes and labels
    for i, prediction in enumerate(results['predictions']):
        if prediction['age'] is not None:
            x1, y1, x2, y2 = prediction['face_box']
            
            # Draw rectangle
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text
            text = f"Age: {prediction['age']} ({prediction['age_range']})"
            confidence_text = f"Conf: {prediction['confidence']:.2f}"
            
            # Draw text background
            cv2.rectangle(display_image, (x1, y1-40), (x2, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(display_image, text, (x1+5, y1-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(display_image, confidence_text, (x1+5, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Convert BGR to RGB for display
    display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    
    # Display the image
    st.image(display_image_rgb, caption=f"Age Detection Results - {results['faces_detected']} faces detected", 
             use_column_width=True)

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">👶 Modern Age Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📸 Single Image", "📁 Batch Processing", "📊 Analytics", "⚙️ Settings"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📸 Single Image":
        show_single_image_page()
    elif page == "📁 Batch Processing":
        show_batch_processing_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    """Display the home page"""
    st.markdown("""
    ## Welcome to the Modern Age Detection System! 🎯
    
    This advanced system uses state-of-the-art deep learning models to accurately predict age from facial images.
    
    ### Features:
    - **Multiple Detection Methods**: DeepFace, MediaPipe, and OpenCV DNN
    - **Ensemble Predictions**: Combines multiple models for improved accuracy
    - **Real-time Processing**: Fast face detection and age estimation
    - **Batch Processing**: Process multiple images at once
    - **Analytics Dashboard**: Track performance and statistics
    - **Database Storage**: Store and analyze detection results
    
    ### How to Use:
    1. **Single Image**: Upload an image and get instant age predictions
    2. **Batch Processing**: Upload multiple images for bulk processing
    3. **Analytics**: View statistics and performance metrics
    4. **Settings**: Configure detection parameters
    """)
    
    # System status
    st.markdown("### System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.detector:
            st.success("✅ Age Detector Loaded")
        else:
            st.error("❌ Age Detector Failed")
    
    with col2:
        stats = get_database_stats()
        st.metric("Total Detections", stats.get('total_detections', 0))
    
    with col3:
        avg_conf = stats.get('average_confidence', 0)
        st.metric("Avg Confidence", f"{avg_conf:.2f}")

def show_single_image_page():
    """Display the single image processing page"""
    st.header("📸 Single Image Age Detection")
    
    if st.session_state.detector is None:
        st.error("Age detector not available. Please check the system status.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing faces for age detection"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("🔍 Detect Age", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    # Convert PIL to OpenCV format
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        cv2.imwrite(tmp_file.name, image_cv)
                        
                        # Process image
                        results = st.session_state.detector.process_image(tmp_file.name)
                        
                        # Clean up
                        os.unlink(tmp_file.name)
                    
                    # Display results
                    st.subheader("Detection Results")
                    
                    if results.get('error'):
                        st.error(f"Error: {results['error']}")
                    else:
                        st.success(f"✅ Successfully detected {results['faces_detected']} face(s)")
                        
                        # Display image with results
                        display_image_with_results(image_cv, results)
                        
                        # Show detailed results
                        st.subheader("Detailed Results")
                        for i, prediction in enumerate(results['predictions']):
                            with st.expander(f"Face {i+1} Details"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Age", prediction['age'])
                                
                                with col2:
                                    st.metric("Age Range", prediction['age_range'])
                                
                                with col3:
                                    st.metric("Confidence", f"{prediction['confidence']:.2f}")
                                
                                # Show method details
                                st.write(f"**Detection Method:** {prediction['method']}")
                                
                                if 'individual_predictions' in prediction:
                                    st.write("**Individual Predictions:**")
                                    for pred in prediction['individual_predictions']:
                                        st.write(f"- {pred['method']}: Age {pred['age']} (Conf: {pred['confidence']:.2f})")
                        
                        # Store results
                        st.session_state.processed_images.append({
                            'filename': uploaded_file.name,
                            'results': results,
                            'timestamp': datetime.now()
                        })
                
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    logger.error(f"Error processing image: {e}")

def show_batch_processing_page():
    """Display the batch processing page"""
    st.header("📁 Batch Processing")
    
    if st.session_state.detector is None:
        st.error("Age detector not available. Please check the system status.")
        return
    
    # File upload for multiple images
    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        help="Upload multiple images for batch processing"
    )
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        # Process all button
        if st.button("🚀 Process All Images", type="primary"):
            progress_bar = st.progress(0)
            results_summary = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Process image
                    image = Image.open(uploaded_file)
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        cv2.imwrite(tmp_file.name, image_cv)
                        
                        # Process image
                        results = st.session_state.detector.process_image(tmp_file.name)
                        
                        # Clean up
                        os.unlink(tmp_file.name)
                    
                    # Store summary
                    results_summary.append({
                        'filename': uploaded_file.name,
                        'faces_detected': results['faces_detected'],
                        'avg_age': np.mean([p['age'] for p in results['predictions'] if p['age'] is not None]) if results['predictions'] else None,
                        'avg_confidence': np.mean([p['confidence'] for p in results['predictions']]) if results['predictions'] else 0,
                        'error': results.get('error')
                    })
                
                except Exception as e:
                    results_summary.append({
                        'filename': uploaded_file.name,
                        'faces_detected': 0,
                        'avg_age': None,
                        'avg_confidence': 0,
                        'error': str(e)
                    })
            
            # Display summary
            st.subheader("Batch Processing Results")
            
            # Create DataFrame
            df = pd.DataFrame(results_summary)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Images", len(df))
            
            with col2:
                successful = len(df[df['faces_detected'] > 0])
                st.metric("Successful", successful)
            
            with col3:
                total_faces = df['faces_detected'].sum()
                st.metric("Total Faces", total_faces)
            
            with col4:
                avg_conf = df[df['avg_confidence'] > 0]['avg_confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.2f}" if not pd.isna(avg_conf) else "N/A")
            
            # Results table
            st.subheader("Detailed Results")
            st.dataframe(df, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"age_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def show_analytics_page():
    """Display the analytics page"""
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.detector is None:
        st.error("Age detector not available. Please check the system status.")
        return
    
    # Get statistics
    stats = get_database_stats()
    
    if not stats:
        st.warning("No data available for analytics.")
        return
    
    # Overview metrics
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Detections", stats.get('total_detections', 0))
    
    with col2:
        st.metric("Average Confidence", f"{stats.get('average_confidence', 0):.2f}")
    
    with col3:
        method_dist = stats.get('method_distribution', {})
        st.metric("Detection Methods", len(method_dist))
    
    with col4:
        # Calculate success rate (assuming confidence > 0.5 as successful)
        conn = sqlite3.connect(st.session_state.detector.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM detections WHERE confidence > 0.5')
        successful_detections = cursor.fetchone()[0]
        success_rate = (successful_detections / stats.get('total_detections', 1)) * 100
        conn.close()
        
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Method distribution pie chart
        if method_dist:
            fig_pie = px.pie(
                values=list(method_dist.values()),
                names=list(method_dist.keys()),
                title="Detection Method Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence distribution
        conn = sqlite3.connect(st.session_state.detector.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT confidence FROM detections WHERE confidence > 0')
        confidences = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if confidences:
            fig_hist = px.histogram(
                x=confidences,
                title="Confidence Score Distribution",
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Age range distribution
    conn = sqlite3.connect(st.session_state.detector.db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT age_range, COUNT(*) FROM detections GROUP BY age_range')
    age_data = cursor.fetchall()
    conn.close()
    
    if age_data:
        st.subheader("Age Range Distribution")
        age_df = pd.DataFrame(age_data, columns=['Age Range', 'Count'])
        fig_bar = px.bar(
            age_df,
            x='Age Range',
            y='Count',
            title="Detected Age Ranges"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Recent detections table
    st.subheader("Recent Detections")
    conn = sqlite3.connect(st.session_state.detector.db_path)
    recent_df = pd.read_sql_query('''
        SELECT image_path, detected_age, age_range, confidence, method, timestamp
        FROM detections 
        ORDER BY timestamp DESC 
        LIMIT 20
    ''', conn)
    conn.close()
    
    if not recent_df.empty:
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No recent detections found.")

def show_settings_page():
    """Display the settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("Detection Parameters")
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Minimum Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence score for accepting age predictions"
    )
    
    # Detection method preference
    method_preference = st.selectbox(
        "Preferred Detection Method",
        ["ensemble", "deepface", "heuristic"],
        help="Choose the primary detection method"
    )
    
    # Database management
    st.subheader("Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Data"):
            if st.session_state.detector:
                conn = sqlite3.connect(st.session_state.detector.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM detections')
                conn.commit()
                conn.close()
                st.success("All detection data cleared!")
    
    with col2:
        if st.button("📊 Generate Sample Data"):
            if st.session_state.detector:
                st.session_state.detector.create_sample_database()
                st.success("Sample data generated!")
    
    # Export data
    st.subheader("Data Export")
    
    if st.button("📥 Export All Data"):
        if st.session_state.detector:
            conn = sqlite3.connect(st.session_state.detector.db_path)
            df = pd.read_sql_query('SELECT * FROM detections', conn)
            conn.close()
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"age_detection_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # System information
    st.subheader("System Information")
    
    info_data = {
        "Python Version": "3.8+",
        "OpenCV Version": cv2.__version__,
        "Streamlit Version": st.__version__,
        "Device": "CUDA" if torch.cuda.is_available() else "CPU",
        "Database Path": st.session_state.detector.db_path if st.session_state.detector else "N/A"
    }
    
    for key, value in info_data.items():
        st.write(f"**{key}:** {value}")

if __name__ == "__main__":
    main()
