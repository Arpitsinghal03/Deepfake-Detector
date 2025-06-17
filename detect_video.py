import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Union
import face_recognition
from .face_analysis import analyze_face_manipulation

class VideoAnalyzer:
    def __init__(self, model_path: Union[str, Path] = None):
        """
        Initialize the video analyzer with a deepfake detection model.
        
        Args:
            model_path: Path to the pre-trained model file
        """
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path:
            self.load_model(model_path)
        
        # Initialize face detection parameters
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def load_model(self, model_path: Union[str, Path]):
        """
        Load a pre-trained deepfake detection model.
        
        Args:
            model_path: Path to the model file
        """
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        
        try:
            self.model = torch.load(str(model_path), map_location=self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a video frame for model input.
        
        Args:
            frame: Input video frame
            
        Returns:
            torch.Tensor: Preprocessed frame
        """
        # Resize frame to model input size
        frame = cv2.resize(frame, (224, 224))
        
        # Convert to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PyTorch tensor and normalize
        frame = torch.from_numpy(frame).float()
        frame = frame.permute(2, 0, 1)  # Change from HWC to CHW format
        frame = frame / 255.0  # Normalize to [0, 1]
        
        return frame.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in a video frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List[Dict]: List of detected faces with their locations
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to list of dictionaries
        face_list = []
        for i, (x, y, w, h) in enumerate(faces):
            face_list.append({
                'id': i,
                'bbox': (x, y, w, h),
                'frame': frame[y:y+h, x:x+w]
            })
        
        return face_list
    
    def analyze_video(self, video_path: Union[str, Path]) -> Dict:
        """
        Analyze a video file for deepfake detection.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict: Analysis results including deepfake probability and face analysis
        """
        if not isinstance(video_path, Path):
            video_path = Path(video_path)
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Initialize results
        frame_predictions = []
        face_analysis_results = []
        
        # Process video frames
        frame_count = 0
        with torch.no_grad():  # Disable gradient computation
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 5th frame to save computation
                if frame_count % 5 == 0:
                    # Detect faces
                    faces = self.detect_faces(frame)
                    
                    # Analyze each face
                    for face in faces:
                        face_result = analyze_face_manipulation(face['frame'])
                        face_analysis_results.append({
                            'id': face['id'],
                            'frame': frame_count,
                            'manipulation_score': face_result['manipulation_score'],
                            'features': face_result['features']
                        })
                    
                    # Get frame prediction if model is loaded
                    if self.model is not None:
                        processed_frame = self.preprocess_frame(frame)
                        prediction = torch.sigmoid(self.model(processed_frame)).item()
                        frame_predictions.append(prediction)
                
                frame_count += 1
        
        # Release video capture
        cap.release()
        
        # Calculate final results
        if frame_predictions:
            deepfake_probability = np.mean(frame_predictions)
            confidence_score = 1 - np.std(frame_predictions)
        else:
            deepfake_probability = 0.5  # Default if no model predictions
            confidence_score = 0.0
        
        # Aggregate face analysis results
        face_analysis = []
        for face_id in set(f['id'] for f in face_analysis_results):
            face_frames = [f for f in face_analysis_results if f['id'] == face_id]
            avg_manipulation = np.mean([f['manipulation_score'] for f in face_frames])
            face_analysis.append({
                'id': face_id,
                'manipulation_score': float(avg_manipulation),
                'frames_analyzed': len(face_frames)
            })
        
        return {
            'deepfake_probability': float(deepfake_probability),
            'confidence_score': float(confidence_score),
            'face_analysis': face_analysis,
            'frames_analyzed': frame_count
        }

def analyze_video(video_path: Union[str, Path], model_path: Union[str, Path] = None) -> Dict:
    """
    Convenience function to analyze a video file.
    
    Args:
        video_path: Path to the video file
        model_path: Optional path to the model file
        
    Returns:
        Dict: Analysis results
    """
    analyzer = VideoAnalyzer(model_path)
    return analyzer.analyze_video(video_path) 