"""
BE FAST Scale Implementation
Comprehensive stroke detection using Balance, Eyes, Face, Arms, Speech, and Time assessment
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from datetime import datetime
import json


class TestResult(Enum):
    NORMAL = "Normal"
    ABNORMAL = "Abnormal"
    INCONCLUSIVE = "Inconclusive"


class RiskLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class TestComponent:
    """Individual BE FAST test component"""
    name: str
    result: TestResult
    score: float  # 0-1 scale
    details: Dict
    timestamp: str


@dataclass
class BEFASTAssessment:
    """Complete BE FAST assessment results"""
    balance: TestComponent
    eyes: TestComponent
    face: TestComponent
    arms: TestComponent
    speech: TestComponent
    time: TestComponent
    overall_score: float
    risk_level: RiskLevel
    recommendations: List[str]
    timestamp: str


class BalanceTest:
    """Balance assessment using computer vision"""
    
    def __init__(self):
        self.pose_analyzer = PoseAnalyzer()
    
    def assess_balance(self, video_frames: List[np.ndarray]) -> TestComponent:
        """Assess balance using pose estimation"""
        if not video_frames:
            return TestComponent(
                name="Balance",
                result=TestResult.INCONCLUSIVE,
                score=0.0,
                details={"error": "No video frames provided"},
                timestamp=datetime.now().isoformat()
            )
        
        # Analyze pose stability across frames
        stability_score = self._calculate_stability(video_frames)
        
        # Determine result based on stability
        if stability_score > 0.8:
            result = TestResult.NORMAL
        elif stability_score > 0.5:
            result = TestResult.INCONCLUSIVE
        else:
            result = TestResult.ABNORMAL
        
        return TestComponent(
            name="Balance",
            result=result,
            score=stability_score,
            details={
                "stability_score": stability_score,
                "frames_analyzed": len(video_frames),
                "assessment_method": "pose_stability"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_stability(self, frames: List[np.ndarray]) -> float:
        """Calculate balance stability from video frames"""
        if len(frames) < 2:
            return 0.0
        
        # Extract key points from each frame
        key_points = []
        for frame in frames:
            points = self.pose_analyzer.extract_key_points(frame)
            if points:
                key_points.append(points)
        
        if len(key_points) < 2:
            return 0.0
        
        # Calculate center of mass movement
        com_movements = []
        for i in range(1, len(key_points)):
            prev_com = self._calculate_center_of_mass(key_points[i-1])
            curr_com = self._calculate_center_of_mass(key_points[i])
            
            if prev_com and curr_com:
                movement = np.linalg.norm(np.array(curr_com) - np.array(prev_com))
                com_movements.append(movement)
        
        if not com_movements:
            return 0.0
        
        # Calculate stability score (lower movement = higher stability)
        avg_movement = np.mean(com_movements)
        max_expected_movement = 50  # pixels
        
        stability_score = max(0, 1 - (avg_movement / max_expected_movement))
        return min(1.0, stability_score)
    
    def _calculate_center_of_mass(self, key_points: Dict) -> Optional[Tuple[float, float]]:
        """Calculate center of mass from key points"""
        if not key_points:
            return None
        
        # Use shoulder and hip points for center of mass
        shoulder_points = []
        hip_points = []
        
        for point_name, (x, y, conf) in key_points.items():
            if 'shoulder' in point_name.lower() and conf > 0.5:
                shoulder_points.append((x, y))
            elif 'hip' in point_name.lower() and conf > 0.5:
                hip_points.append((x, y))
        
        if shoulder_points and hip_points:
            shoulder_center = np.mean(shoulder_points, axis=0)
            hip_center = np.mean(hip_points, axis=0)
            com = (shoulder_center + hip_center) / 2
            return tuple(com)
        
        return None


class EyeTrackingTest:
    """Eye tracking assessment for gaze deviation detection"""
    
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def assess_eye_tracking(self, video_frames: List[np.ndarray], target_positions: List[Tuple[int, int]]) -> TestComponent:
        """Assess eye tracking ability"""
        if not video_frames or not target_positions:
            return TestComponent(
                name="Eyes",
                result=TestResult.INCONCLUSIVE,
                score=0.0,
                details={"error": "Insufficient data for eye tracking assessment"},
                timestamp=datetime.now().isoformat()
            )
        
        # Track eye movement accuracy
        tracking_accuracy = self._calculate_tracking_accuracy(video_frames, target_positions)
        
        # Determine result
        if tracking_accuracy > 0.7:
            result = TestResult.NORMAL
        elif tracking_accuracy > 0.4:
            result = TestResult.INCONCLUSIVE
        else:
            result = TestResult.ABNORMAL
        
        return TestComponent(
            name="Eyes",
            result=result,
            score=tracking_accuracy,
            details={
                "tracking_accuracy": tracking_accuracy,
                "frames_analyzed": len(video_frames),
                "target_positions": len(target_positions),
                "assessment_method": "gaze_tracking"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_tracking_accuracy(self, frames: List[np.ndarray], targets: List[Tuple[int, int]]) -> float:
        """Calculate how accurately eyes track the moving target"""
        if len(frames) != len(targets):
            return 0.0
        
        accuracies = []
        
        for frame, target in zip(frames, targets):
            # Detect eyes in frame
            eyes = self._detect_eyes(frame)
            if not eyes:
                continue
            
            # Calculate gaze direction (simplified)
            gaze_direction = self._estimate_gaze_direction(frame, eyes)
            if gaze_direction:
                # Calculate accuracy based on gaze-target alignment
                target_direction = self._calculate_target_direction(target, frame.shape)
                accuracy = self._calculate_direction_accuracy(gaze_direction, target_direction)
                accuracies.append(accuracy)
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def _detect_eyes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect eyes in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 3)
        return eyes.tolist()
    
    def _estimate_gaze_direction(self, frame: np.ndarray, eyes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[float, float]]:
        """Estimate gaze direction from eye position"""
        if len(eyes) < 2:
            return None
        
        # Sort eyes by x-coordinate
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Calculate center points
        left_eye_center = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
        right_eye_center = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
        
        # Calculate gaze direction (simplified - center between eyes)
        gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2
        gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2
        
        # Normalize to frame coordinates
        frame_height, frame_width = frame.shape[:2]
        normalized_gaze = (gaze_x / frame_width, gaze_y / frame_height)
        
        return normalized_gaze
    
    def _calculate_target_direction(self, target: Tuple[int, int], frame_shape: Tuple[int, int, int]) -> Tuple[float, float]:
        """Calculate normalized target direction"""
        frame_height, frame_width = frame_shape[:2]
        normalized_target = (target[0] / frame_width, target[1] / frame_height)
        return normalized_target
    
    def _calculate_direction_accuracy(self, gaze: Tuple[float, float], target: Tuple[float, float]) -> float:
        """Calculate accuracy between gaze and target directions"""
        distance = np.sqrt((gaze[0] - target[0])**2 + (gaze[1] - target[1])**2)
        # Convert distance to accuracy (closer = more accurate)
        accuracy = max(0, 1 - distance * 2)  # Scale factor for normalization
        return accuracy


class FacialExpressionTest:
    """Facial expression assessment for symmetry detection"""
    
    def __init__(self):
        self.facial_analyzer = FacialAnalyzer()
    
    def assess_facial_expressions(self, images: List[np.ndarray], expressions: List[str]) -> TestComponent:
        """Assess facial expression symmetry"""
        if not images or not expressions:
            return TestComponent(
                name="Face",
                result=TestResult.INCONCLUSIVE,
                score=0.0,
                details={"error": "No images or expressions provided"},
                timestamp=datetime.now().isoformat()
            )
        
        # Analyze each expression
        expression_scores = []
        for image, expression in zip(images, expressions):
            score = self._analyze_expression(image, expression)
            expression_scores.append(score)
        
        # Calculate overall facial symmetry score
        overall_score = np.mean(expression_scores)
        
        # Determine result
        if overall_score > 0.8:
            result = TestResult.NORMAL
        elif overall_score > 0.6:
            result = TestResult.INCONCLUSIVE
        else:
            result = TestResult.ABNORMAL
        
        return TestComponent(
            name="Face",
            result=result,
            score=overall_score,
            details={
                "expression_scores": expression_scores,
                "expressions_tested": expressions,
                "overall_symmetry": overall_score,
                "assessment_method": "facial_symmetry"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _analyze_expression(self, image: np.ndarray, expression: str) -> float:
        """Analyze specific facial expression for symmetry"""
        features = self.facial_analyzer.analyze_facial_features(image)
        if not features:
            return 0.0
        
        # Use overall symmetry as base score
        base_symmetry = features.overall_symmetry
        
        # Adjust based on expression type
        expression_adjustments = {
            "smile": 0.1,  # Smiles should be symmetric
            "frown": 0.1,  # Frowns should be symmetric
            "raise_eyebrows": 0.05,  # Eyebrow raises
            "close_eyes": 0.0,  # Eye closing
            "neutral": 0.0  # Neutral expression
        }
        
        adjustment = expression_adjustments.get(expression.lower(), 0.0)
        adjusted_score = min(1.0, base_symmetry + adjustment)
        
        return adjusted_score


class ArmMovementTest:
    """Arm movement assessment for weakness detection"""
    
    def __init__(self):
        self.pose_analyzer = PoseAnalyzer()
    
    def assess_arm_movement(self, video_frames: List[np.ndarray]) -> TestComponent:
        """Assess arm movement symmetry and strength"""
        if not video_frames:
            return TestComponent(
                name="Arms",
                result=TestResult.INCONCLUSIVE,
                score=0.0,
                details={"error": "No video frames provided"},
                timestamp=datetime.now().isoformat()
            )
        
        # Analyze arm movement patterns
        arm_scores = []
        for frame in video_frames:
            score = self._analyze_arm_position(frame)
            arm_scores.append(score)
        
        overall_score = np.mean(arm_scores)
        
        # Determine result
        if overall_score > 0.8:
            result = TestResult.NORMAL
        elif overall_score > 0.6:
            result = TestResult.INCONCLUSIVE
        else:
            result = TestResult.ABNORMAL
        
        return TestComponent(
            name="Arms",
            result=result,
            score=overall_score,
            details={
                "arm_scores": arm_scores,
                "frames_analyzed": len(video_frames),
                "overall_symmetry": overall_score,
                "assessment_method": "arm_movement_analysis"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _analyze_arm_position(self, frame: np.ndarray) -> float:
        """Analyze arm position for symmetry"""
        key_points = self.pose_analyzer.extract_key_points(frame)
        if not key_points:
            return 0.0
        
        # Extract arm key points
        left_arm_points = []
        right_arm_points = []
        
        for point_name, (x, y, conf) in key_points.items():
            if conf > 0.5:  # Only use confident detections
                if 'left' in point_name.lower() and ('shoulder' in point_name.lower() or 'elbow' in point_name.lower() or 'wrist' in point_name.lower()):
                    left_arm_points.append((x, y))
                elif 'right' in point_name.lower() and ('shoulder' in point_name.lower() or 'elbow' in point_name.lower() or 'wrist' in point_name.lower()):
                    right_arm_points.append((x, y))
        
        if len(left_arm_points) < 2 or len(right_arm_points) < 2:
            return 0.0
        
        # Calculate arm angles
        left_angle = self._calculate_arm_angle(left_arm_points)
        right_angle = self._calculate_arm_angle(right_arm_points)
        
        if left_angle is None or right_angle is None:
            return 0.0
        
        # Calculate symmetry score
        angle_diff = abs(left_angle - right_angle)
        max_expected_diff = 30  # degrees
        
        symmetry_score = max(0, 1 - (angle_diff / max_expected_diff))
        return symmetry_score
    
    def _calculate_arm_angle(self, arm_points: List[Tuple[float, float]]) -> Optional[float]:
        """Calculate arm angle from key points"""
        if len(arm_points) < 2:
            return None
        
        # Sort points by y-coordinate (top to bottom)
        arm_points = sorted(arm_points, key=lambda p: p[1])
        
        # Calculate angle between shoulder and wrist
        shoulder = arm_points[0]
        wrist = arm_points[-1]
        
        # Calculate angle
        dx = wrist[0] - shoulder[0]
        dy = wrist[1] - shoulder[1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle


class SpeechAnalysisTest:
    """Speech analysis for dysarthria detection"""
    
    def __init__(self):
        self.speech_analyzer = SpeechAnalyzer()
    
    def assess_speech(self, audio_data: np.ndarray, text: str) -> TestComponent:
        """Assess speech clarity and fluency"""
        if audio_data is None or not text:
            return TestComponent(
                name="Speech",
                result=TestResult.INCONCLUSIVE,
                score=0.0,
                details={"error": "No audio data or text provided"},
                timestamp=datetime.now().isoformat()
            )
        
        # Analyze speech characteristics
        speech_features = self.speech_analyzer.analyze_speech(audio_data)
        
        # Calculate speech quality score
        quality_score = self._calculate_speech_quality(speech_features)
        
        # Determine result
        if quality_score > 0.7:
            result = TestResult.NORMAL
        elif quality_score > 0.4:
            result = TestResult.INCONCLUSIVE
        else:
            result = TestResult.ABNORMAL
        
        return TestComponent(
            name="Speech",
            result=result,
            score=quality_score,
            details={
                "speech_features": speech_features,
                "quality_score": quality_score,
                "text_length": len(text),
                "assessment_method": "speech_analysis"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_speech_quality(self, features: Dict) -> float:
        """Calculate overall speech quality score"""
        # Weight different speech characteristics
        weights = {
            'clarity': 0.3,
            'fluency': 0.3,
            'pace': 0.2,
            'volume': 0.2
        }
        
        quality_score = 0.0
        for feature, weight in weights.items():
            if feature in features:
                quality_score += features[feature] * weight
        
        return min(1.0, quality_score)


class TimeTrackingTest:
    """Time tracking for symptom onset documentation"""
    
    def __init__(self):
        self.start_time = None
        self.test_times = {}
    
    def start_test(self) -> None:
        """Start timing the test"""
        self.start_time = datetime.now()
    
    def record_test_time(self, test_name: str) -> None:
        """Record time for specific test"""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            self.test_times[test_name] = elapsed.total_seconds()
    
    def assess_timing(self) -> TestComponent:
        """Assess timing and create time component"""
        if not self.start_time:
            return TestComponent(
                name="Time",
                result=TestResult.INCONCLUSIVE,
                score=0.0,
                details={"error": "Test timing not started"},
                timestamp=datetime.now().isoformat()
            )
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        # Score based on completion time (faster is better for stroke detection)
        if total_time < 60:  # Less than 1 minute
            score = 1.0
        elif total_time < 120:  # Less than 2 minutes
            score = 0.8
        elif total_time < 180:  # Less than 3 minutes
            score = 0.6
        else:
            score = 0.4
        
        return TestComponent(
            name="Time",
            result=TestResult.NORMAL if score > 0.6 else TestResult.ABNORMAL,
            score=score,
            details={
                "total_time_seconds": total_time,
                "test_times": self.test_times,
                "start_time": self.start_time.isoformat(),
                "assessment_method": "timing_analysis"
            },
            timestamp=datetime.now().isoformat()
        )


class BEFASTAssessmentEngine:
    """Main engine for BE FAST assessment"""
    
    def __init__(self):
        self.balance_test = BalanceTest()
        self.eye_test = EyeTrackingTest()
        self.face_test = FacialExpressionTest()
        self.arm_test = ArmMovementTest()
        self.speech_test = SpeechAnalysisTest()
        self.time_test = TimeTrackingTest()
    
    def perform_assessment(self, 
                          balance_video: List[np.ndarray] = None,
                          eye_tracking_data: Tuple[List[np.ndarray], List[Tuple[int, int]]] = None,
                          facial_images: Tuple[List[np.ndarray], List[str]] = None,
                          arm_video: List[np.ndarray] = None,
                          speech_data: Tuple[np.ndarray, str] = None) -> BEFASTAssessment:
        """Perform complete BE FAST assessment"""
        
        # Start timing
        self.time_test.start_test()
        
        # Perform individual tests
        balance = self.balance_test.assess_balance(balance_video or [])
        self.time_test.record_test_time("balance")
        
        eyes = self.eye_test.assess_eye_tracking(
            eye_tracking_data[0] if eye_tracking_data else [],
            eye_tracking_data[1] if eye_tracking_data else []
        )
        self.time_test.record_test_time("eyes")
        
        face = self.face_test.assess_facial_expressions(
            facial_images[0] if facial_images else [],
            facial_images[1] if facial_images else []
        )
        self.time_test.record_test_time("face")
        
        arms = self.arm_test.assess_arm_movement(arm_video or [])
        self.time_test.record_test_time("arms")
        
        speech = self.speech_test.assess_speech(
            speech_data[0] if speech_data else None,
            speech_data[1] if speech_data else ""
        )
        self.time_test.record_test_time("speech")
        
        time_component = self.time_test.assess_timing()
        
        # Calculate overall score and risk level
        overall_score = self._calculate_overall_score([balance, eyes, face, arms, speech, time_component])
        risk_level = self._determine_risk_level(overall_score)
        recommendations = self._generate_recommendations(risk_level, [balance, eyes, face, arms, speech])
        
        return BEFASTAssessment(
            balance=balance,
            eyes=eyes,
            face=face,
            arms=arms,
            speech=speech,
            time=time_component,
            overall_score=overall_score,
            risk_level=risk_level,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_overall_score(self, components: List[TestComponent]) -> float:
        """Calculate overall BE FAST score"""
        # Weight different components
        weights = {
            'Balance': 0.15,
            'Eyes': 0.15,
            'Face': 0.25,
            'Arms': 0.25,
            'Speech': 0.15,
            'Time': 0.05
        }
        
        weighted_score = 0.0
        for component in components:
            weight = weights.get(component.name, 0.0)
            weighted_score += component.score * weight
        
        return weighted_score
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from overall score"""
        if score >= 0.8:
            return RiskLevel.LOW
        elif score >= 0.6:
            return RiskLevel.MODERATE
        elif score >= 0.4:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_recommendations(self, risk_level: RiskLevel, components: List[TestComponent]) -> List[str]:
        """Generate recommendations based on risk level and component results"""
        recommendations = []
        
        # Base recommendations by risk level
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "IMMEDIATE MEDICAL ATTENTION REQUIRED",
                "Call emergency services (911) immediately",
                "Do not drive to hospital - call ambulance",
                "Time is critical for stroke treatment"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "High risk detected - seek medical evaluation immediately",
                "Contact your doctor or visit emergency room",
                "Monitor symptoms closely",
                "Prepare emergency contact information"
            ])
        elif risk_level == RiskLevel.MODERATE:
            recommendations.extend([
                "Moderate risk detected - monitor closely",
                "Schedule appointment with healthcare provider",
                "Repeat test if symptoms worsen",
                "Maintain healthy lifestyle habits"
            ])
        else:  # LOW
            recommendations.extend([
                "Low risk detected - continue monitoring",
                "Maintain regular health checkups",
                "Continue healthy lifestyle habits",
                "Repeat assessment monthly"
            ])
        
        # Add specific recommendations based on failed components
        for component in components:
            if component.result == TestResult.ABNORMAL:
                if component.name == "Face":
                    recommendations.append("Facial asymmetry detected - monitor for facial drooping")
                elif component.name == "Arms":
                    recommendations.append("Arm weakness detected - monitor for arm drift")
                elif component.name == "Speech":
                    recommendations.append("Speech difficulties detected - monitor for slurred speech")
                elif component.name == "Eyes":
                    recommendations.append("Eye movement issues detected - monitor for vision problems")
                elif component.name == "Balance":
                    recommendations.append("Balance issues detected - monitor for coordination problems")
        
        return recommendations


# Helper classes (simplified implementations)
class PoseAnalyzer:
    """Simplified pose analysis for balance and arm movement"""
    
    def extract_key_points(self, frame: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """Extract key body points from frame"""
        # Simplified implementation - in real app would use MediaPipe or similar
        return {
            'left_shoulder': (100, 150, 0.8),
            'right_shoulder': (200, 150, 0.8),
            'left_elbow': (120, 200, 0.7),
            'right_elbow': (180, 200, 0.7),
            'left_wrist': (140, 250, 0.6),
            'right_wrist': (160, 250, 0.6),
            'left_hip': (130, 300, 0.8),
            'right_hip': (170, 300, 0.8)
        }


class SpeechAnalyzer:
    """Simplified speech analysis for dysarthria detection"""
    
    def analyze_speech(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze speech characteristics"""
        # Simplified implementation - in real app would use advanced audio processing
        return {
            'clarity': 0.8,
            'fluency': 0.7,
            'pace': 0.9,
            'volume': 0.8
        }


class FacialAnalyzer:
    """Simplified facial analyzer"""
    
    def analyze_facial_features(self, image: np.ndarray) -> Optional[object]:
        """Analyze facial features"""
        # Simplified implementation - would use actual facial analysis
        class FacialFeatures:
            def __init__(self):
                self.overall_symmetry = 0.8
        
        return FacialFeatures()
