import os
import cv2
import numpy as np
import sqlite3
import requests
from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MockDatabaseGenerator:
    """Generate mock database with sample images and age labels"""
    
    def __init__(self, output_dir="sample_images"):
        self.output_dir = output_dir
        self.db_path = "age_detection.db"
        os.makedirs(output_dir, exist_ok=True)
        
        # Age categories with sample counts
        self.age_categories = {
            "0-2": {"count": 5, "description": "Infants and toddlers"},
            "3-9": {"count": 8, "description": "Children"},
            "10-19": {"count": 10, "description": "Teenagers"},
            "20-29": {"count": 12, "description": "Young adults"},
            "30-39": {"count": 10, "description": "Adults"},
            "40-49": {"count": 8, "description": "Middle-aged adults"},
            "50-59": {"count": 6, "description": "Mature adults"},
            "60-69": {"count": 4, "description": "Senior adults"},
            "70+": {"count": 3, "description": "Elderly adults"}
        }
    
    def generate_synthetic_faces(self, age_range: str, count: int):
        """Generate synthetic face images for testing"""
        generated_images = []
        
        for i in range(count):
            # Create a simple synthetic face
            img = self._create_synthetic_face(age_range, i)
            
            # Save image
            filename = f"synthetic_{age_range.replace('-', '_')}_{i+1}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath)
            
            generated_images.append({
                "filepath": filepath,
                "age_range": age_range,
                "true_age": self._get_random_age_in_range(age_range),
                "source": "synthetic"
            })
        
        return generated_images
    
    def _create_synthetic_face(self, age_range: str, index: int):
        """Create a simple synthetic face image"""
        # Create a blank image
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Face parameters based on age
        age_params = self._get_age_parameters(age_range)
        
        # Draw face outline
        face_size = age_params["face_size"]
        face_x = (200 - face_size) // 2
        face_y = (200 - face_size) // 2
        
        draw.ellipse([face_x, face_y, face_x + face_size, face_y + face_size], 
                    fill='peachpuff', outline='black', width=2)
        
        # Draw eyes
        eye_y = face_y + face_size // 3
        left_eye_x = face_x + face_size // 4
        right_eye_x = face_x + 3 * face_size // 4
        
        draw.ellipse([left_eye_x - 5, eye_y - 5, left_eye_x + 5, eye_y + 5], 
                    fill='black')
        draw.ellipse([right_eye_x - 5, eye_y - 5, right_eye_x + 5, eye_y + 5], 
                    fill='black')
        
        # Draw nose
        nose_x = face_x + face_size // 2
        nose_y = face_y + face_size // 2
        draw.polygon([(nose_x, nose_y - 3), (nose_x - 3, nose_y + 3), 
                     (nose_x + 3, nose_y + 3)], fill='peachpuff', outline='black')
        
        # Draw mouth
        mouth_y = face_y + 2 * face_size // 3
        mouth_width = age_params["mouth_width"]
        draw.arc([nose_x - mouth_width//2, mouth_y - 5, 
                 nose_x + mouth_width//2, mouth_y + 5], 
                 start=0, end=180, fill='black', width=2)
        
        # Add age-specific features
        if age_range in ["0-2", "3-9"]:
            # Add freckles for children
            for _ in range(random.randint(2, 5)):
                x = random.randint(face_x + 10, face_x + face_size - 10)
                y = random.randint(face_y + 10, face_y + face_size - 10)
                draw.ellipse([x-1, y-1, x+1, y+1], fill='brown')
        
        elif age_range in ["60-69", "70+"]:
            # Add wrinkles for elderly
            for _ in range(random.randint(3, 6)):
                start_x = random.randint(face_x + 10, face_x + face_size - 10)
                start_y = random.randint(face_y + 10, face_y + face_size - 10)
                end_x = start_x + random.randint(-10, 10)
                end_y = start_y + random.randint(-5, 5)
                draw.line([start_x, start_y, end_x, end_y], fill='gray', width=1)
        
        return img
    
    def _get_age_parameters(self, age_range: str):
        """Get face parameters based on age range"""
        params = {
            "0-2": {"face_size": 120, "mouth_width": 15},
            "3-9": {"face_size": 130, "mouth_width": 18},
            "10-19": {"face_size": 140, "mouth_width": 20},
            "20-29": {"face_size": 150, "mouth_width": 22},
            "30-39": {"face_size": 150, "mouth_width": 22},
            "40-49": {"face_size": 145, "mouth_width": 20},
            "50-59": {"face_size": 140, "mouth_width": 18},
            "60-69": {"face_size": 135, "mouth_width": 16},
            "70+": {"face_size": 130, "mouth_width": 14}
        }
        return params.get(age_range, params["20-29"])
    
    def _get_random_age_in_range(self, age_range: str):
        """Get a random age within the specified range"""
        if age_range == "0-2":
            return random.randint(0, 2)
        elif age_range == "3-9":
            return random.randint(3, 9)
        elif age_range == "10-19":
            return random.randint(10, 19)
        elif age_range == "20-29":
            return random.randint(20, 29)
        elif age_range == "30-39":
            return random.randint(30, 39)
        elif age_range == "40-49":
            return random.randint(40, 49)
        elif age_range == "50-59":
            return random.randint(50, 59)
        elif age_range == "60-69":
            return random.randint(60, 69)
        elif age_range == "70+":
            return random.randint(70, 85)
        else:
            return random.randint(20, 50)
    
    def download_sample_images(self):
        """Download sample images from public sources"""
        # This is a placeholder for downloading real sample images
        # In a real implementation, you would download from appropriate sources
        logger.info("Sample image download functionality would be implemented here")
        return []
    
    def create_mock_database(self):
        """Create the complete mock database"""
        logger.info("Creating mock database with sample images...")
        
        all_images = []
        
        # Generate synthetic images for each age category
        for age_range, config in self.age_categories.items():
            logger.info(f"Generating {config['count']} images for age range {age_range}")
            images = self.generate_synthetic_faces(age_range, config['count'])
            all_images.extend(images)
        
        # Create database entries
        self._populate_database(all_images)
        
        logger.info(f"Mock database created with {len(all_images)} sample images")
        return all_images
    
    def _populate_database(self, images):
        """Populate the database with image information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sample_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                true_age INTEGER,
                age_range TEXT,
                source TEXT,
                added_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                detected_age INTEGER,
                age_range TEXT,
                confidence REAL,
                method TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                face_count INTEGER
            )
        ''')
        
        # Insert sample images
        for img_data in images:
            cursor.execute('''
                INSERT OR IGNORE INTO sample_images 
                (image_path, true_age, age_range, source)
                VALUES (?, ?, ?, ?)
            ''', (
                img_data["filepath"],
                img_data["true_age"],
                img_data["age_range"],
                img_data["source"]
            ))
        
        conn.commit()
        conn.close()
    
    def get_database_stats(self):
        """Get statistics from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get sample image statistics
        cursor.execute('SELECT COUNT(*) FROM sample_images')
        total_samples = cursor.fetchone()[0]
        
        cursor.execute('SELECT age_range, COUNT(*) FROM sample_images GROUP BY age_range')
        age_distribution = dict(cursor.fetchall())
        
        cursor.execute('SELECT COUNT(*) FROM detections')
        total_detections = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_samples": total_samples,
            "age_distribution": age_distribution,
            "total_detections": total_detections
        }

def create_sample_video():
    """Create a sample video for testing video processing"""
    logger.info("Creating sample video...")
    
    # Create a simple video with moving synthetic faces
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('sample_video.mp4', fourcc, 20.0, (640, 480))
    
    generator = MockDatabaseGenerator()
    
    for frame_num in range(100):  # 5 seconds at 20 fps
        # Create frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.fill(255)  # White background
        
        # Add moving face
        face_img = generator._create_synthetic_face("20-29", frame_num)
        face_array = np.array(face_img)
        
        # Resize and position
        face_resized = cv2.resize(face_array, (100, 100))
        
        # Move face across screen
        x_pos = (frame_num * 6) % 540
        y_pos = 190
        
        frame[y_pos:y_pos+100, x_pos:x_pos+100] = face_resized
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    logger.info("Sample video created: sample_video.mp4")

def main():
    """Main function to create mock database"""
    generator = MockDatabaseGenerator()
    
    print("Creating mock database for age detection system...")
    print("=" * 50)
    
    # Create sample images
    images = generator.create_mock_database()
    
    # Get statistics
    stats = generator.get_database_stats()
    
    print(f"✅ Created {stats['total_samples']} sample images")
    print(f"📊 Age distribution: {stats['age_distribution']}")
    print(f"🗄️ Total detections in database: {stats['total_detections']}")
    
    # Create sample video
    create_sample_video()
    
    print("\n🎬 Sample video created: sample_video.mp4")
    print("📁 Sample images saved in: sample_images/")
    print("🗄️ Database file: age_detection.db")

if __name__ == "__main__":
    main()
