"""
Advanced Camera Integration for StrokeSense
Real-time camera processing, pose estimation, and eye tracking
"""

import cv2
import numpy as np
import threading
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import queue
import mediapipe as mp
from models.ml_models import MLModelManager

class CameraState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PROCESSING = "processing"
    ERROR = "error"

@dataclass
class CameraFrame:
    """Camera frame with metadata"""
    frame: np.ndarray
    timestamp: float
    frame_id: int
    landmarks: Optional[Dict] = None
    analysis_results: Optional[Dict] = None

@dataclass
class EyeTrackingData:
    """Eye tracking data"""
    left_eye_center: Tuple[float, float]
    right_eye_center: Tuple[float, float]
    gaze_direction: Tuple[float, float]
    blink_detected: bool
    eye_aspect_ratio: float
    confidence: float

@dataclass
class PoseData:
    """Pose estimation data"""
    landmarks: List[Tuple[float, float, float]]
    center_of_mass: Tuple[float, float]
    stability_score: float
    arm_positions: Dict[str, Tuple[float, float]]
    confidence: float

class AdvancedCameraProcessor:
    """Advanced camera processing with real-time analysis"""
    
    def __init__(self):
        self.camera = None
        self.state = CameraState.IDLE
        self.frame_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.stop_processing = False
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_detector = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize ML models
        self.ml_manager = MLModelManager()
        
        # Eye tracking parameters
        self.eye_landmarks = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        }
        
        # Callbacks
        self.frame_callbacks = []
        self.analysis_callbacks = []
        
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """Initialize camera with error handling"""
        try:
            self.state = CameraState.INITIALIZING
            
            # Try to open camera
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                # Try different camera indices
                for i in range(1, 5):
                    self.camera = cv2.VideoCapture(i)
                    if self.camera.isOpened():
                        break
                
                if not self.camera.isOpened():
                    raise Exception("No camera found")
            
            # Configure camera settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Camera test failed")
            
            self.state = CameraState.ACTIVE
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            self.state = CameraState.ERROR
            return False
    
    def start_processing(self):
        """Start camera processing thread"""
        if self.state != CameraState.ACTIVE:
            return False
        
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return True
    
    def stop_camera(self):
        """Stop camera and processing"""
        self.stop_processing = True
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.state = CameraState.IDLE
    
    def _processing_loop(self):
        """Main processing loop"""
        frame_id = 0
        
        while not self.stop_processing and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                frame_id += 1
                timestamp = time.time()
                
                # Create frame object
                camera_frame = CameraFrame(
                    frame=frame.copy(),
                    timestamp=timestamp,
                    frame_id=frame_id
                )
                
                # Process frame
                self._process_frame(camera_frame)
                
                # Add to queue
                try:
                    self.frame_queue.put_nowait(camera_frame)
                except queue.Full:
                    # Remove oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(camera_frame)
                    except queue.Empty:
                        pass
                
                # Notify callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(camera_frame)
                    except Exception as e:
                        print(f"Frame callback error: {e}")
                
                # Control frame rate
                time.sleep(1/30)  # 30 FPS
                
            except Exception as e:
                print(f"Processing loop error: {e}")
                break
    
    def _process_frame(self, camera_frame: CameraFrame):
        """Process individual frame for analysis"""
        try:
            frame = camera_frame.frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect pose
            pose_results = self.pose_detector.process(rgb_frame)
            pose_data = self._extract_pose_data(pose_results, frame.shape)
            
            # Detect face and eyes
            face_results = self.face_detector.process(rgb_frame)
            eye_data = self._extract_eye_data(face_results, frame.shape)
            
            # Store results
            camera_frame.landmarks = {
                'pose': pose_data,
                'eyes': eye_data
            }
            
            # Perform ML analysis
            analysis_results = self._analyze_frame(frame, pose_data, eye_data)
            camera_frame.analysis_results = analysis_results
            
            # Notify analysis callbacks
            for callback in self.analysis_callbacks:
                try:
                    callback(camera_frame)
                except Exception as e:
                    print(f"Analysis callback error: {e}")
                    
        except Exception as e:
            print(f"Frame processing error: {e}")
    
    def _extract_pose_data(self, pose_results, frame_shape) -> Optional[PoseData]:
        """Extract pose data from MediaPipe results"""
        if not pose_results.pose_landmarks:
            return None
        
        landmarks = []
        for landmark in pose_results.pose_landmarks.landmark:
            landmarks.append((
                landmark.x * frame_shape[1],  # Convert to pixel coordinates
                landmark.y * frame_shape[0],
                landmark.z
            ))
        
        # Calculate center of mass
        center_x = np.mean([lm[0] for lm in landmarks])
        center_y = np.mean([lm[1] for lm in landmarks])
        
        # Extract arm positions
        arm_positions = {
            'left_shoulder': landmarks[11],
            'right_shoulder': landmarks[12],
            'left_elbow': landmarks[13],
            'right_elbow': landmarks[14],
            'left_wrist': landmarks[15],
            'right_wrist': landmarks[16]
        }
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(landmarks)
        
        return PoseData(
            landmarks=landmarks,
            center_of_mass=(center_x, center_y),
            stability_score=stability_score,
            arm_positions=arm_positions,
            confidence=pose_results.pose_landmarks.visibility
        )
    
    def _extract_eye_data(self, face_results, frame_shape) -> Optional[EyeTrackingData]:
        """Extract eye tracking data from MediaPipe results"""
        if not face_results.multi_face_landmarks:
            return None
        
        face_landmarks = face_results.multi_face_landmarks[0]
        
        # Extract eye landmarks
        left_eye_points = []
        right_eye_points = []
        
        for idx in self.eye_landmarks['left_eye']:
            landmark = face_landmarks.landmark[idx]
            left_eye_points.append((
                landmark.x * frame_shape[1],
                landmark.y * frame_shape[0]
            ))
        
        for idx in self.eye_landmarks['right_eye']:
            landmark = face_landmarks.landmark[idx]
            right_eye_points.append((
                landmark.x * frame_shape[1],
                landmark.y * frame_shape[0]
            ))
        
        # Calculate eye centers
        left_eye_center = (
            np.mean([p[0] for p in left_eye_points]),
            np.mean([p[1] for p in left_eye_points])
        )
        right_eye_center = (
            np.mean([p[0] for p in right_eye_points]),
            np.mean([p[1] for p in right_eye_points])
        )
        
        # Calculate gaze direction
        gaze_direction = (
            (left_eye_center[0] + right_eye_center[0]) / 2,
            (left_eye_center[1] + right_eye_center[1]) / 2
        )
        
        # Calculate Eye Aspect Ratio (EAR) for blink detection
        left_ear = self._calculate_ear(left_eye_points)
        right_ear = self._calculate_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2
        
        # Detect blink (simplified)
        blink_detected = avg_ear < 0.25
        
        return EyeTrackingData(
            left_eye_center=left_eye_center,
            right_eye_center=right_eye_center,
            gaze_direction=gaze_direction,
            blink_detected=blink_detected,
            eye_aspect_ratio=avg_ear,
            confidence=0.8  # MediaPipe confidence
        )
    
    def _calculate_ear(self, eye_points: List[Tuple[float, float]]) -> float:
        """Calculate Eye Aspect Ratio"""
        if len(eye_points) < 6:
            return 0.0
        
        # Vertical distances
        vertical_1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        vertical_2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        
        # Horizontal distance
        horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def _calculate_stability_score(self, landmarks: List[Tuple[float, float, float]]) -> float:
        """Calculate balance stability score"""
        if len(landmarks) < 33:  # MediaPipe pose has 33 landmarks
            return 0.0
        
        # Use key points for stability calculation
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Calculate center of mass
        com_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
        com_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4
        
        # Calculate base of support
        base_center_x = (left_ankle[0] + right_ankle[0]) / 2
        base_center_y = (left_ankle[1] + right_ankle[1]) / 2
        
        # Calculate stability (distance from COM to base center)
        stability_distance = np.sqrt((com_x - base_center_x)**2 + (com_y - base_center_y)**2)
        
        # Normalize to stability score (lower distance = higher stability)
        max_expected_distance = 100  # pixels
        stability_score = max(0, 1 - (stability_distance / max_expected_distance))
        
        return min(1.0, stability_score)
    
    def _analyze_frame(self, frame: np.ndarray, pose_data: Optional[PoseData], 
                      eye_data: Optional[EyeTrackingData]) -> Dict[str, Any]:
        """Perform ML analysis on frame"""
        analysis_results = {
            'timestamp': time.time(),
            'facial_analysis': None,
            'pose_analysis': None,
            'eye_analysis': None
        }
        
        try:
            # Facial analysis
            facial_results = self.ml_manager.analyze_facial_symmetry(frame)
            analysis_results['facial_analysis'] = facial_results
            
            # Pose analysis
            if pose_data:
                pose_results = self.ml_manager.analyze_balance(frame)
                analysis_results['pose_analysis'] = {
                    'stability_score': pose_data.stability_score,
                    'ml_stability_score': pose_results['stability_score'],
                    'center_of_mass': pose_data.center_of_mass,
                    'arm_positions': pose_data.arm_positions,
                    'confidence': pose_data.confidence
                }
            
            # Eye analysis
            if eye_data:
                analysis_results['eye_analysis'] = {
                    'gaze_direction': eye_data.gaze_direction,
                    'blink_detected': eye_data.blink_detected,
                    'eye_aspect_ratio': eye_data.eye_aspect_ratio,
                    'confidence': eye_data.confidence
                }
            
        except Exception as e:
            print(f"Frame analysis error: {e}")
        
        return analysis_results
    
    def add_frame_callback(self, callback: Callable[[CameraFrame], None]):
        """Add callback for new frames"""
        self.frame_callbacks.append(callback)
    
    def add_analysis_callback(self, callback: Callable[[CameraFrame], None]):
        """Add callback for analysis results"""
        self.analysis_callbacks.append(callback)
    
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Get latest processed frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information"""
        if not self.camera:
            return {'status': 'Not initialized'}
        
        return {
            'status': self.state.value,
            'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.camera.get(cv2.CAP_PROP_FPS)),
            'queue_size': self.frame_queue.qsize(),
            'processing_active': self.processing_thread and self.processing_thread.is_alive()
        }


class EyeTrackingTest:
    """Eye tracking test implementation"""
    
    def __init__(self, camera_processor: AdvancedCameraProcessor):
        self.camera_processor = camera_processor
        self.test_active = False
        self.target_positions = []
        self.tracking_data = []
        self.test_results = {}
    
    def start_eye_tracking_test(self, duration: float = 10.0) -> bool:
        """Start eye tracking test"""
        if self.test_active:
            return False
        
        self.test_active = True
        self.tracking_data = []
        self.target_positions = self._generate_target_positions(duration)
        
        # Add analysis callback
        self.camera_processor.add_analysis_callback(self._on_frame_analysis)
        
        return True
    
    def stop_eye_tracking_test(self) -> Dict[str, Any]:
        """Stop test and return results"""
        if not self.test_active:
            return {}
        
        self.test_active = False
        
        # Calculate tracking accuracy
        accuracy = self._calculate_tracking_accuracy()
        
        self.test_results = {
            'tracking_accuracy': accuracy,
            'data_points': len(self.tracking_data),
            'test_duration': len(self.tracking_data) / 30.0,  # Assuming 30 FPS
            'blink_count': sum(1 for data in self.tracking_data if data['blink_detected']),
            'average_ear': np.mean([data['eye_aspect_ratio'] for data in self.tracking_data])
        }
        
        return self.test_results
    
    def _generate_target_positions(self, duration: float) -> List[Tuple[float, float]]:
        """Generate target positions for eye tracking"""
        positions = []
        num_positions = int(duration * 2)  # Change position every 0.5 seconds
        
        for i in range(num_positions):
            # Generate positions in a pattern
            angle = (i / num_positions) * 2 * np.pi
            radius = 100  # pixels from center
            
            x = 320 + radius * np.cos(angle)  # Center around 320px
            y = 240 + radius * np.sin(angle)  # Center around 240px
            
            positions.append((x, y))
        
        return positions
    
    def _on_frame_analysis(self, camera_frame: CameraFrame):
        """Handle frame analysis during test"""
        if not self.test_active:
            return
        
        if camera_frame.analysis_results and camera_frame.analysis_results['eye_analysis']:
            eye_data = camera_frame.analysis_results['eye_analysis']
            
            tracking_point = {
                'timestamp': camera_frame.timestamp,
                'gaze_direction': eye_data['gaze_direction'],
                'blink_detected': eye_data['blink_detected'],
                'eye_aspect_ratio': eye_data['eye_aspect_ratio'],
                'confidence': eye_data['confidence']
            }
            
            self.tracking_data.append(tracking_point)
    
    def _calculate_tracking_accuracy(self) -> float:
        """Calculate how well eyes tracked the targets"""
        if not self.tracking_data or not self.target_positions:
            return 0.0
        
        accuracies = []
        
        for i, tracking_point in enumerate(self.tracking_data):
            if i < len(self.target_positions):
                target = self.target_positions[i]
                gaze = tracking_point['gaze_direction']
                
                # Calculate distance between gaze and target
                distance = np.sqrt((gaze[0] - target[0])**2 + (gaze[1] - target[1])**2)
                
                # Convert distance to accuracy (closer = more accurate)
                accuracy = max(0, 1 - (distance / 200))  # Normalize by expected screen size
                accuracies.append(accuracy)
        
        return np.mean(accuracies) if accuracies else 0.0


class PoseTrackingTest:
    """Pose tracking test implementation"""
    
    def __init__(self, camera_processor: AdvancedCameraProcessor):
        self.camera_processor = camera_processor
        self.test_active = False
        self.pose_data = []
        self.test_results = {}
    
    def start_pose_tracking_test(self, duration: float = 10.0) -> bool:
        """Start pose tracking test"""
        if self.test_active:
            return False
        
        self.test_active = True
        self.pose_data = []
        
        # Add analysis callback
        self.camera_processor.add_analysis_callback(self._on_frame_analysis)
        
        return True
    
    def stop_pose_tracking_test(self) -> Dict[str, Any]:
        """Stop test and return results"""
        if not self.test_active:
            return {}
        
        self.test_active = False
        
        # Calculate stability metrics
        stability_scores = [data['stability_score'] for data in self.pose_data]
        arm_symmetry = self._calculate_arm_symmetry()
        
        self.test_results = {
            'average_stability': np.mean(stability_scores),
            'stability_variance': np.var(stability_scores),
            'arm_symmetry_score': arm_symmetry,
            'data_points': len(self.pose_data),
            'test_duration': len(self.pose_data) / 30.0,
            'balance_detected': np.mean(stability_scores) > 0.7
        }
        
        return self.test_results
    
    def _on_frame_analysis(self, camera_frame: CameraFrame):
        """Handle frame analysis during test"""
        if not self.test_active:
            return
        
        if (camera_frame.analysis_results and 
            camera_frame.analysis_results['pose_analysis']):
            
            pose_data = camera_frame.analysis_results['pose_analysis']
            
            tracking_point = {
                'timestamp': camera_frame.timestamp,
                'stability_score': pose_data['stability_score'],
                'center_of_mass': pose_data['center_of_mass'],
                'arm_positions': pose_data['arm_positions'],
                'confidence': pose_data['confidence']
            }
            
            self.pose_data.append(tracking_point)
    
    def _calculate_arm_symmetry(self) -> float:
        """Calculate arm symmetry score"""
        if not self.pose_data:
            return 0.0
        
        symmetry_scores = []
        
        for data in self.pose_data:
            arm_positions = data['arm_positions']
            
            # Calculate arm angles
            left_shoulder = arm_positions['left_shoulder']
            left_elbow = arm_positions['left_elbow']
            left_wrist = arm_positions['left_wrist']
            
            right_shoulder = arm_positions['right_shoulder']
            right_elbow = arm_positions['right_elbow']
            right_wrist = arm_positions['right_wrist']
            
            # Calculate arm vectors
            left_arm_vector = np.array(left_wrist[:2]) - np.array(left_shoulder[:2])
            right_arm_vector = np.array(right_wrist[:2]) - np.array(right_shoulder[:2])
            
            # Calculate angles
            left_angle = np.arctan2(left_arm_vector[1], left_arm_vector[0])
            right_angle = np.arctan2(right_arm_vector[1], right_arm_vector[0])
            
            # Calculate symmetry
            angle_diff = abs(left_angle - right_angle)
            symmetry_score = max(0, 1 - (angle_diff / np.pi))
            symmetry_scores.append(symmetry_score)
        
        return np.mean(symmetry_scores) if symmetry_scores else 0.0

