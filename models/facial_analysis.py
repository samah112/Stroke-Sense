"""
Facial Dimension Analysis System
Creates baseline facial models and detects asymmetry for stroke detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import math


@dataclass
class FacialLandmark:
    """Represents a facial landmark point"""
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class FacialFeatures:
    """Comprehensive facial feature analysis"""
    landmarks: List[FacialLandmark]
    eye_aspect_ratio: float
    mouth_aspect_ratio: float
    eyebrow_symmetry: float
    cheek_symmetry: float
    overall_symmetry: float
    timestamp: str


class FacialAnalyzer:
    """Advanced facial analysis for stroke detection"""
    
    def __init__(self):
        # Initialize face detection models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Facial landmark indices for key features
        self.LANDMARK_INDICES = {
            'left_eye': [36, 37, 38, 39, 40, 41],
            'right_eye': [42, 43, 44, 45, 46, 47],
            'left_eyebrow': [17, 18, 19, 20, 21],
            'right_eyebrow': [22, 23, 24, 25, 26],
            'nose': [27, 28, 29, 30, 31, 32, 33, 34, 35],
            'mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            'jaw': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        }
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in image and return bounding box"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Return the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            return largest_face
        
        return None
    
    def extract_landmarks(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> List[FacialLandmark]:
        """Extract facial landmarks using computer vision techniques"""
        x, y, w, h = face_box
        face_roi = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        landmarks = []
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3)
        
        # Create simplified landmark points based on detected features
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate (left to right)
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Left eye landmarks
            left_eye = eyes[0]
            landmarks.extend([
                FacialLandmark(x + left_eye[0], y + left_eye[1], 0.9),  # Top-left
                FacialLandmark(x + left_eye[0] + left_eye[2], y + left_eye[1], 0.9),  # Top-right
                FacialLandmark(x + left_eye[0] + left_eye[2]//2, y + left_eye[1] + left_eye[3]//2, 0.9),  # Center
                FacialLandmark(x + left_eye[0], y + left_eye[1] + left_eye[3], 0.9),  # Bottom-left
                FacialLandmark(x + left_eye[0] + left_eye[2], y + left_eye[1] + left_eye[3], 0.9),  # Bottom-right
            ])
            
            # Right eye landmarks
            right_eye = eyes[1]
            landmarks.extend([
                FacialLandmark(x + right_eye[0], y + right_eye[1], 0.9),  # Top-left
                FacialLandmark(x + right_eye[0] + right_eye[2], y + right_eye[1], 0.9),  # Top-right
                FacialLandmark(x + right_eye[0] + right_eye[2]//2, y + right_eye[1] + right_eye[3]//2, 0.9),  # Center
                FacialLandmark(x + right_eye[0], y + right_eye[1] + right_eye[3], 0.9),  # Bottom-left
                FacialLandmark(x + right_eye[0] + right_eye[2], y + right_eye[1] + right_eye[3], 0.9),  # Bottom-right
            ])
        
        # Estimate mouth position (center of face, below eyes)
        mouth_y = y + h * 0.7
        mouth_x = x + w // 2
        mouth_width = w * 0.4
        
        landmarks.extend([
            FacialLandmark(mouth_x - mouth_width//2, mouth_y, 0.8),  # Left corner
            FacialLandmark(mouth_x + mouth_width//2, mouth_y, 0.8),  # Right corner
            FacialLandmark(mouth_x, mouth_y + h*0.05, 0.8),  # Center top
            FacialLandmark(mouth_x, mouth_y - h*0.05, 0.8),  # Center bottom
        ])
        
        # Estimate eyebrow positions
        eyebrow_y = y + h * 0.25
        landmarks.extend([
            FacialLandmark(x + w*0.2, eyebrow_y, 0.7),  # Left eyebrow
            FacialLandmark(x + w*0.8, eyebrow_y, 0.7),  # Right eyebrow
        ])
        
        return landmarks
    
    def calculate_eye_aspect_ratio(self, landmarks: List[FacialLandmark]) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        if len(landmarks) < 10:  # Need at least 5 points per eye
            return 0.0
        
        # Left eye EAR
        left_eye_points = landmarks[:5]
        left_ear = self._calculate_ear_for_eye(left_eye_points)
        
        # Right eye EAR
        right_eye_points = landmarks[5:10]
        right_ear = self._calculate_ear_for_eye(right_eye_points)
        
        return (left_ear + right_ear) / 2.0
    
    def _calculate_ear_for_eye(self, eye_points: List[FacialLandmark]) -> float:
        """Calculate EAR for a single eye"""
        if len(eye_points) < 5:
            return 0.0
        
        # Vertical distances
        vertical_1 = self._euclidean_distance(eye_points[1], eye_points[5]) if len(eye_points) > 5 else 0
        vertical_2 = self._euclidean_distance(eye_points[2], eye_points[4])
        
        # Horizontal distance
        horizontal = self._euclidean_distance(eye_points[0], eye_points[3])
        
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def calculate_mouth_aspect_ratio(self, landmarks: List[FacialLandmark]) -> float:
        """Calculate Mouth Aspect Ratio for mouth opening detection"""
        if len(landmarks) < 14:  # Need mouth landmarks
            return 0.0
        
        mouth_points = landmarks[10:14]  # Assuming mouth points are at indices 10-13
        
        # Vertical distance
        vertical = self._euclidean_distance(mouth_points[2], mouth_points[3])
        
        # Horizontal distance
        horizontal = self._euclidean_distance(mouth_points[0], mouth_points[1])
        
        if horizontal == 0:
            return 0.0
        
        mar = vertical / horizontal
        return mar
    
    def calculate_symmetry_score(self, landmarks: List[FacialLandmark]) -> Dict[str, float]:
        """Calculate various symmetry scores for facial features"""
        symmetry_scores = {
            'eye_symmetry': 0.0,
            'eyebrow_symmetry': 0.0,
            'mouth_symmetry': 0.0,
            'overall_symmetry': 0.0
        }
        
        if len(landmarks) < 14:
            return symmetry_scores
        
        # Eye symmetry (compare left and right eye positions)
        if len(landmarks) >= 10:
            left_eye_center = self._get_center_point(landmarks[:5])
            right_eye_center = self._get_center_point(landmarks[5:10])
            symmetry_scores['eye_symmetry'] = self._calculate_feature_symmetry(left_eye_center, right_eye_center)
        
        # Eyebrow symmetry
        if len(landmarks) >= 12:
            left_eyebrow = landmarks[10]
            right_eyebrow = landmarks[11]
            symmetry_scores['eyebrow_symmetry'] = self._calculate_feature_symmetry(left_eyebrow, right_eyebrow)
        
        # Mouth symmetry
        if len(landmarks) >= 14:
            mouth_left = landmarks[12]
            mouth_right = landmarks[13]
            symmetry_scores['mouth_symmetry'] = self._calculate_feature_symmetry(mouth_left, mouth_right)
        
        # Overall symmetry (weighted average)
        weights = [0.3, 0.2, 0.2, 0.3]  # eye, eyebrow, mouth, overall
        symmetry_scores['overall_symmetry'] = sum(
            score * weight for score, weight in zip(
                [symmetry_scores['eye_symmetry'], symmetry_scores['eyebrow_symmetry'], 
                 symmetry_scores['mouth_symmetry'], symmetry_scores['mouth_symmetry']], 
                weights
            )
        )
        
        return symmetry_scores
    
    def _euclidean_distance(self, point1: FacialLandmark, point2: FacialLandmark) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def _get_center_point(self, points: List[FacialLandmark]) -> FacialLandmark:
        """Calculate center point of a group of landmarks"""
        if not points:
            return FacialLandmark(0, 0, 0)
        
        avg_x = sum(p.x for p in points) / len(points)
        avg_y = sum(p.y for p in points) / len(points)
        avg_confidence = sum(p.confidence for p in points) / len(points)
        
        return FacialLandmark(avg_x, avg_y, avg_confidence)
    
    def _calculate_feature_symmetry(self, left_point: FacialLandmark, right_point: FacialLandmark) -> float:
        """Calculate symmetry score between left and right features"""
        # Calculate vertical alignment (should be similar)
        vertical_diff = abs(left_point.y - right_point.y)
        
        # Calculate horizontal distance from center
        center_x = (left_point.x + right_point.x) / 2
        left_distance = abs(left_point.x - center_x)
        right_distance = abs(right_point.x - center_x)
        
        # Symmetry score (higher is more symmetric)
        vertical_score = max(0, 1 - vertical_diff / 50)  # Normalize by expected face size
        horizontal_score = max(0, 1 - abs(left_distance - right_distance) / 50)
        
        return (vertical_score + horizontal_score) / 2.0
    
    def analyze_facial_features(self, image: np.ndarray) -> Optional[FacialFeatures]:
        """Comprehensive facial feature analysis"""
        face_box = self.detect_face(image)
        if not face_box:
            return None
        
        landmarks = self.extract_landmarks(image, face_box)
        if not landmarks:
            return None
        
        # Calculate various metrics
        eye_aspect_ratio = self.calculate_eye_aspect_ratio(landmarks)
        mouth_aspect_ratio = self.calculate_mouth_aspect_ratio(landmarks)
        symmetry_scores = self.calculate_symmetry_score(landmarks)
        
        return FacialFeatures(
            landmarks=landmarks,
            eye_aspect_ratio=eye_aspect_ratio,
            mouth_aspect_ratio=mouth_aspect_ratio,
            eyebrow_symmetry=symmetry_scores['eyebrow_symmetry'],
            cheek_symmetry=symmetry_scores['eye_symmetry'],  # Using eye symmetry as proxy
            overall_symmetry=symmetry_scores['overall_symmetry'],
            timestamp=str(np.datetime64('now'))
        )


class BaselineModel:
    """Manages baseline facial models for comparison"""
    
    def __init__(self):
        self.baseline_features: Optional[FacialFeatures] = None
        self.analyzer = FacialAnalyzer()
    
    def create_baseline(self, image: np.ndarray) -> bool:
        """Create baseline facial model from healthy state image"""
        features = self.analyzer.analyze_facial_features(image)
        if features and features.overall_symmetry > 0.7:  # Only accept good quality baselines
            self.baseline_features = features
            return True
        return False
    
    def compare_with_baseline(self, current_features: FacialFeatures) -> Dict[str, float]:
        """Compare current features with baseline to detect changes"""
        if not self.baseline_features:
            return {'error': 'No baseline model available'}
        
        comparison = {
            'symmetry_change': abs(current_features.overall_symmetry - self.baseline_features.overall_symmetry),
            'eye_aspect_change': abs(current_features.eye_aspect_ratio - self.baseline_features.eye_aspect_ratio),
            'mouth_aspect_change': abs(current_features.mouth_aspect_ratio - self.baseline_features.mouth_aspect_ratio),
            'eyebrow_symmetry_change': abs(current_features.eyebrow_symmetry - self.baseline_features.eyebrow_symmetry),
            'cheek_symmetry_change': abs(current_features.cheek_symmetry - self.baseline_features.cheek_symmetry)
        }
        
        # Calculate overall deviation score
        deviation_scores = list(comparison.values())[:-1]  # Exclude error if present
        comparison['overall_deviation'] = sum(deviation_scores) / len(deviation_scores)
        
        return comparison
    
    def detect_facial_weakness(self, current_features: FacialFeatures) -> Dict[str, any]:
        """Detect potential facial weakness or asymmetry"""
        comparison = self.compare_with_baseline(current_features)
        
        if 'error' in comparison:
            return comparison
        
        # Thresholds for detecting potential issues
        thresholds = {
            'symmetry_threshold': 0.15,
            'eye_aspect_threshold': 0.1,
            'mouth_aspect_threshold': 0.2,
            'overall_deviation_threshold': 0.2
        }
        
        alerts = []
        risk_score = 0
        
        if comparison['symmetry_change'] > thresholds['symmetry_threshold']:
            alerts.append("Significant facial asymmetry detected")
            risk_score += 2
        
        if comparison['eye_aspect_change'] > thresholds['eye_aspect_threshold']:
            alerts.append("Eye movement irregularity detected")
            risk_score += 1
        
        if comparison['mouth_aspect_change'] > thresholds['mouth_aspect_threshold']:
            alerts.append("Mouth movement irregularity detected")
            risk_score += 2
        
        if comparison['overall_deviation'] > thresholds['overall_deviation_threshold']:
            alerts.append("Overall facial movement deviation detected")
            risk_score += 3
        
        return {
            'alerts': alerts,
            'risk_score': risk_score,
            'comparison_data': comparison,
            'recommendation': self._get_recommendation(risk_score)
        }
    
    def _get_recommendation(self, risk_score: int) -> str:
        """Get recommendation based on risk score"""
        if risk_score == 0:
            return "No significant facial changes detected"
        elif risk_score <= 2:
            return "Minor changes detected. Monitor closely and repeat test if symptoms persist"
        elif risk_score <= 4:
            return "Moderate changes detected. Consider medical evaluation"
        else:
            return "Significant changes detected. Immediate medical attention recommended"
    
    def save_baseline(self, filepath: str) -> bool:
        """Save baseline model to file"""
        if not self.baseline_features:
            return False
        
        try:
            baseline_data = {
                'landmarks': [(lm.x, lm.y, lm.confidence) for lm in self.baseline_features.landmarks],
                'eye_aspect_ratio': self.baseline_features.eye_aspect_ratio,
                'mouth_aspect_ratio': self.baseline_features.mouth_aspect_ratio,
                'eyebrow_symmetry': self.baseline_features.eyebrow_symmetry,
                'cheek_symmetry': self.baseline_features.cheek_symmetry,
                'overall_symmetry': self.baseline_features.overall_symmetry,
                'timestamp': self.baseline_features.timestamp
            }
            
            with open(filepath, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving baseline: {e}")
            return False
    
    def load_baseline(self, filepath: str) -> bool:
        """Load baseline model from file"""
        try:
            with open(filepath, 'r') as f:
                baseline_data = json.load(f)
            
            landmarks = [FacialLandmark(x, y, conf) for x, y, conf in baseline_data['landmarks']]
            
            self.baseline_features = FacialFeatures(
                landmarks=landmarks,
                eye_aspect_ratio=baseline_data['eye_aspect_ratio'],
                mouth_aspect_ratio=baseline_data['mouth_aspect_ratio'],
                eyebrow_symmetry=baseline_data['eyebrow_symmetry'],
                cheek_symmetry=baseline_data['cheek_symmetry'],
                overall_symmetry=baseline_data['overall_symmetry'],
                timestamp=baseline_data['timestamp']
            )
            
            return True
        except Exception as e:
            print(f"Error loading baseline: {e}")
            return False
