import cv2
import numpy as np
from typing import Dict, List, Tuple
import face_recognition
from scipy import signal

def analyze_face_manipulation(face_image: np.ndarray) -> Dict:
    """
    Analyze a face image for signs of manipulation.
    
    Args:
        face_image: Input face image
        
    Returns:
        Dict: Analysis results including manipulation score and features
    """
    # Ensure image is in RGB format
    if len(face_image.shape) == 2:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
    elif face_image.shape[2] == 4:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)
    elif face_image.shape[2] == 3:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Extract features
    features = extract_face_features(face_image)
    
    # Calculate manipulation score based on features
    manipulation_score = calculate_manipulation_score(features)
    
    return {
        'manipulation_score': float(manipulation_score),
        'features': features
    }

def extract_face_features(face_image: np.ndarray) -> Dict:
    """
    Extract various features from a face image that might indicate manipulation.
    
    Args:
        face_image: Input face image
        
    Returns:
        Dict: Extracted features
    """
    features = {}
    
    # 1. Face landmarks
    try:
        landmarks = face_recognition.face_landmarks(face_image)
        if landmarks:
            features['landmarks'] = analyze_landmarks(landmarks[0])
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        features['landmarks'] = None
    
    # 2. Frequency domain analysis
    features['frequency'] = analyze_frequency_domain(face_image)
    
    # 3. Color analysis
    features['color'] = analyze_color_distribution(face_image)
    
    # 4. Edge analysis
    features['edges'] = analyze_edges(face_image)
    
    # 5. Texture analysis
    features['texture'] = analyze_texture(face_image)
    
    return features

def analyze_landmarks(landmarks: Dict) -> Dict:
    """
    Analyze facial landmarks for unnatural patterns.
    
    Args:
        landmarks: Dictionary of facial landmarks
        
    Returns:
        Dict: Landmark analysis results
    """
    results = {}
    
    # Calculate symmetry scores for different facial features
    if 'left_eye' in landmarks and 'right_eye' in landmarks:
        results['eye_symmetry'] = calculate_symmetry(
            landmarks['left_eye'],
            landmarks['right_eye']
        )
    
    if 'left_eyebrow' in landmarks and 'right_eyebrow' in landmarks:
        results['eyebrow_symmetry'] = calculate_symmetry(
            landmarks['left_eyebrow'],
            landmarks['right_eyebrow']
        )
    
    if 'top_lip' in landmarks and 'bottom_lip' in landmarks:
        results['lip_symmetry'] = calculate_symmetry(
            landmarks['top_lip'],
            landmarks['bottom_lip']
        )
    
    return results

def calculate_symmetry(points1: List[Tuple[int, int]], points2: List[Tuple[int, int]]) -> float:
    """
    Calculate symmetry score between two sets of points.
    
    Args:
        points1: First set of points
        points2: Second set of points
        
    Returns:
        float: Symmetry score (0-1, higher means more symmetric)
    """
    if len(points1) != len(points2):
        return 0.0
    
    # Calculate center points
    center1 = np.mean(points1, axis=0)
    center2 = np.mean(points2, axis=0)
    
    # Calculate distances from centers
    dist1 = np.array([np.linalg.norm(p - center1) for p in points1])
    dist2 = np.array([np.linalg.norm(p - center2) for p in points2])
    
    # Calculate symmetry score
    symmetry_score = 1 - np.mean(np.abs(dist1 - dist2)) / np.mean(dist1)
    return float(max(0, min(1, symmetry_score)))

def analyze_frequency_domain(image: np.ndarray) -> Dict:
    """
    Analyze image in frequency domain for signs of manipulation.
    
    Args:
        image: Input image
        
    Returns:
        Dict: Frequency domain analysis results
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply 2D FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Analyze frequency distribution
    results = {
        'high_freq_energy': float(np.mean(magnitude_spectrum[100:, 100:])),
        'low_freq_energy': float(np.mean(magnitude_spectrum[:100, :100])),
        'frequency_ratio': float(np.mean(magnitude_spectrum[100:, 100:]) / 
                               (np.mean(magnitude_spectrum[:100, :100]) + 1e-6))
    }
    
    return results

def analyze_color_distribution(image: np.ndarray) -> Dict:
    """
    Analyze color distribution for signs of manipulation.
    
    Args:
        image: Input image
        
    Returns:
        Dict: Color analysis results
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Calculate color statistics
    results = {
        'hue_std': float(np.std(hsv[:, :, 0])),
        'saturation_std': float(np.std(hsv[:, :, 1])),
        'value_std': float(np.std(hsv[:, :, 2])),
        'color_consistency': float(np.mean([
            np.std(hsv[:, :, 0]),
            np.std(hsv[:, :, 1]),
            np.std(hsv[:, :, 2])
        ]))
    }
    
    return results

def analyze_edges(image: np.ndarray) -> Dict:
    """
    Analyze edge patterns for signs of manipulation.
    
    Args:
        image: Input image
        
    Returns:
        Dict: Edge analysis results
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Calculate edge statistics
    results = {
        'edge_density': float(np.mean(edges > 0)),
        'edge_regularity': float(np.std(edges[edges > 0])),
        'edge_direction': analyze_edge_direction(edges)
    }
    
    return results

def analyze_edge_direction(edges: np.ndarray) -> float:
    """
    Analyze edge direction patterns.
    
    Args:
        edges: Edge image
        
    Returns:
        float: Edge direction score
    """
    # Calculate gradients
    sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate angles
    angles = np.arctan2(sobely, sobelx)
    
    # Calculate angle histogram
    hist, _ = np.histogram(angles[edges > 0], bins=36, range=(-np.pi, np.pi))
    
    # Calculate direction score (higher means more regular patterns)
    return float(np.std(hist) / (np.mean(hist) + 1e-6))

def analyze_texture(image: np.ndarray) -> Dict:
    """
    Analyze texture patterns for signs of manipulation.
    
    Args:
        image: Input image
        
    Returns:
        Dict: Texture analysis results
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate GLCM (Gray-Level Co-occurrence Matrix) features
    results = {
        'texture_contrast': calculate_texture_contrast(gray),
        'texture_homogeneity': calculate_texture_homogeneity(gray),
        'texture_energy': calculate_texture_energy(gray)
    }
    
    return results

def calculate_texture_contrast(gray: np.ndarray) -> float:
    """Calculate texture contrast."""
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    return float(graycoprops(glcm, 'contrast').mean())

def calculate_texture_homogeneity(gray: np.ndarray) -> float:
    """Calculate texture homogeneity."""
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    return float(graycoprops(glcm, 'homogeneity').mean())

def calculate_texture_energy(gray: np.ndarray) -> float:
    """Calculate texture energy."""
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    return float(graycoprops(glcm, 'energy').mean())

def calculate_manipulation_score(features: Dict) -> float:
    """
    Calculate overall manipulation score from features.
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        float: Manipulation score (0-1, higher means more likely manipulated)
    """
    scores = []
    
    # Landmark analysis
    if features['landmarks']:
        landmark_scores = [
            features['landmarks'].get('eye_symmetry', 0.5),
            features['landmarks'].get('eyebrow_symmetry', 0.5),
            features['landmarks'].get('lip_symmetry', 0.5)
        ]
        scores.append(1 - np.mean(landmark_scores))  # Less symmetry = more likely manipulated
    
    # Frequency analysis
    freq = features['frequency']
    freq_score = freq['frequency_ratio'] / (freq['frequency_ratio'] + 1)
    scores.append(freq_score)
    
    # Color analysis
    color = features['color']
    color_score = color['color_consistency'] / 255.0
    scores.append(color_score)
    
    # Edge analysis
    edges = features['edges']
    edge_score = edges['edge_regularity'] / 255.0
    scores.append(edge_score)
    
    # Texture analysis
    texture = features['texture']
    texture_score = (texture['texture_contrast'] + 
                    texture['texture_homogeneity'] + 
                    texture['texture_energy']) / 3
    scores.append(texture_score)
    
    # Calculate final score
    final_score = np.mean(scores)
    return float(max(0, min(1, final_score))) 