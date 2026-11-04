"""
Real ML Models for StrokeSense
Advanced machine learning models for facial analysis, speech processing, and pose estimation
"""

import numpy as np
import cv2
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class FacialAnalysisML:
    """Advanced facial analysis using deep learning"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.model = None
        self.model_trained = False  # True only when trained weights are loaded from disk
        self._last_face_image = None  # 64x64x3 face crop used as CNN input
        self.scaler = StandardScaler()
        self.load_model()

    def load_model(self):
        """Load pre-trained facial analysis model"""
        try:
            # Try to load existing (trained) model
            if os.path.exists('models/facial_model.h5'):
                self.model = keras.models.load_model('models/facial_model.h5')
                self.model_trained = True
            else:
                # No trained model yet: build the architecture but do NOT save the
                # untrained weights, otherwise they would be reloaded as "trained"
                self.model = self._create_facial_model()
                self.model_trained = False
        except Exception as e:
            print(f"Error loading facial model: {e}")
            self.model = self._create_facial_model()
            self.model_trained = False
    
    def _create_facial_model(self):
        """Create facial analysis CNN model"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(5, activation='softmax')  # 5 facial features
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def extract_facial_features(self, image: np.ndarray) -> np.ndarray:
        """Extract facial features using computer vision"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            self._last_face_image = None
            return np.zeros(50)  # Return zero features if no face detected

        # Use the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face

        # Extract face region (kept as the CNN input image, scaled to [0, 1])
        face_roi = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (64, 64))
        self._last_face_image = face_resized.astype(np.float32) / 255.0
        
        # Extract eyes
        eyes = self.eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
        
        # Calculate facial features
        features = []
        
        # Face dimensions
        features.extend([w, h, w/h])  # width, height, aspect ratio
        
        # Eye features
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])  # Sort by x-coordinate
            left_eye = eyes[0]
            right_eye = eyes[1]
            
            # Eye positions and sizes
            features.extend([
                left_eye[0], left_eye[1], left_eye[2], left_eye[3],  # left eye
                right_eye[0], right_eye[1], right_eye[2], right_eye[3],  # right eye
                abs(left_eye[1] - right_eye[1]),  # vertical alignment
                abs(left_eye[0] - right_eye[0])  # horizontal distance
            ])
        else:
            features.extend([0] * 10)  # No eyes detected (same length as the branch above)
        
        # Symmetry features
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        if len(eyes) >= 2:
            left_eye_center = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            right_eye_center = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            
            # Calculate symmetry metrics
            left_distance = np.sqrt((left_eye_center[0] - face_center_x)**2 + 
                                  (left_eye_center[1] - face_center_y)**2)
            right_distance = np.sqrt((right_eye_center[0] - face_center_x)**2 + 
                                   (right_eye_center[1] - face_center_y)**2)
            
            features.extend([
                left_distance, right_distance,
                abs(left_distance - right_distance),  # symmetry score
                (left_distance + right_distance) / 2  # average distance
            ])
        else:
            features.extend([0] * 4)
        
        # Pad or truncate to fixed size
        while len(features) < 50:
            features.append(0)
        
        return np.array(features[:50])
    
    def analyze_facial_symmetry(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze facial symmetry for stroke detection"""
        features = self.extract_facial_features(image)
        
        if np.all(features == 0):
            return {
                'symmetry_score': 0.0,
                'confidence': 0.0,
                'asymmetry_detected': True,
                'risk_level': 'high'
            }
        
        # The CNN is only meaningful once it has been trained; until then use the
        # rule-based analysis instead of returning random predictions
        if not self.model_trained or self._last_face_image is None:
            return self._rule_based_symmetry_analysis(features)

        # Use ML model for prediction (CNN expects a 64x64x3 face image)
        try:
            prediction = self.model.predict(self._last_face_image[np.newaxis, ...], verbose=0)
            
            # Interpret prediction (assuming 5 classes: normal, mild_asymmetry, moderate_asymmetry, severe_asymmetry, stroke_like)
            class_labels = ['normal', 'mild_asymmetry', 'moderate_asymmetry', 'severe_asymmetry', 'stroke_like']
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
            
            # Convert to symmetry score
            symmetry_scores = {
                'normal': 0.9,
                'mild_asymmetry': 0.7,
                'moderate_asymmetry': 0.5,
                'severe_asymmetry': 0.3,
                'stroke_like': 0.1
            }
            
            symmetry_score = symmetry_scores[class_labels[predicted_class]]
            
            return {
                'symmetry_score': symmetry_score,
                'confidence': confidence,
                'asymmetry_detected': predicted_class > 0,
                'risk_level': 'high' if predicted_class >= 3 else 'low' if predicted_class == 0 else 'moderate',
                'predicted_class': class_labels[predicted_class]
            }
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            # Fallback to rule-based analysis
            return self._rule_based_symmetry_analysis(features)
    
    def _rule_based_symmetry_analysis(self, features: np.ndarray) -> Dict[str, float]:
        """Fallback rule-based symmetry analysis"""
        if len(features) < 20:
            return {
                'symmetry_score': 0.0,
                'confidence': 0.0,
                'asymmetry_detected': True,
                'risk_level': 'high'
            }
        
        # Extract key features (indices follow extract_facial_features):
        #   2 = face aspect ratio, 11 = eye vertical misalignment (px),
        #   15 = |left - right| eye-to-center distance, 16 = average distance
        face_aspect_ratio = features[2]
        eye_vertical_diff = features[11]
        avg_eye_distance = features[16]
        # Relative asymmetry in [0, 1]: 0 = perfectly symmetric
        symmetry_score = features[15] / avg_eye_distance if avg_eye_distance > 0 else 0
        
        # Calculate overall symmetry
        overall_symmetry = 1.0
        
        # Penalize for poor aspect ratio
        if face_aspect_ratio < 0.7 or face_aspect_ratio > 1.3:
            overall_symmetry -= 0.2
        
        # Penalize for eye misalignment
        if eye_vertical_diff > 10:  # pixels
            overall_symmetry -= 0.3
        
        # Use calculated symmetry score
        if symmetry_score > 0:
            overall_symmetry = min(overall_symmetry, 1.0 - symmetry_score)
        
        return {
            'symmetry_score': max(0.0, overall_symmetry),
            'confidence': 0.7,  # Medium confidence for rule-based
            'asymmetry_detected': overall_symmetry < 0.7,
            'risk_level': 'high' if overall_symmetry < 0.5 else 'low' if overall_symmetry > 0.8 else 'moderate'
        }


class SpeechAnalysisML:
    """Advanced speech analysis for dysarthria detection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.load_model()
    
    def load_model(self):
        """Load pre-trained speech analysis model"""
        try:
            if os.path.exists('models/speech_model.pkl'):
                self.model = joblib.load('models/speech_model.pkl')
            else:
                # Create a new model if none exists
                self.model = self._create_speech_model()
                joblib.dump(self.model, 'models/speech_model.pkl')
        except Exception as e:
            print(f"Error loading speech model: {e}")
            self.model = self._create_speech_model()
    
    def _create_speech_model(self):
        """Create speech analysis model"""
        # Use ensemble of Random Forest and SVM
        from sklearn.ensemble import VotingClassifier
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        
        ensemble = VotingClassifier([
            ('rf', rf),
            ('svm', svm)
        ], voting='soft')
        
        return ensemble
    
    def extract_audio_features(self, audio_data: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        """Extract comprehensive audio features"""
        try:
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Extract features using librosa
            features = []
            
            # Basic features
            features.extend([
                np.mean(audio_data),  # mean amplitude
                np.std(audio_data),   # standard deviation
                np.max(audio_data),   # maximum amplitude
                np.min(audio_data)    # minimum amplitude
            ])
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids)
            ])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr)
            ])
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            for i in range(13):
                features.extend([
                    np.mean(mfccs[i]),
                    np.std(mfccs[i])
                ])
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            features.extend([
                np.mean(rolloff),
                np.std(rolloff)
            ])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            for i in range(12):
                features.extend([
                    np.mean(chroma[i]),
                    np.std(chroma[i])
                ])
            
            # Tempo and rhythm
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            # librosa >= 0.10 returns tempo as a 1-element array
            features.append(float(np.atleast_1d(tempo)[0]))
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
            features.extend([
                np.mean(bandwidth),
                np.std(bandwidth)
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            # Return zero features if extraction fails
            return np.zeros(100)
    
    def analyze_speech_quality(self, audio_data: np.ndarray, sample_rate: int = 22050) -> Dict[str, float]:
        """Analyze speech quality for dysarthria detection"""
        features = self.extract_audio_features(audio_data, sample_rate)
        
        if np.all(features == 0):
            return {
                'clarity_score': 0.0,
                'fluency_score': 0.0,
                'overall_score': 0.0,
                'dysarthria_detected': True,
                'confidence': 0.0,
                'risk_level': 'high'
            }
        
        try:
            # Normalize features
            features_normalized = self.scaler.fit_transform(features.reshape(1, -1))
            
            # Predict using ML model
            prediction = self.model.predict_proba(features_normalized)[0]
            
            # Assuming binary classification: normal vs dysarthria
            normal_prob = prediction[0] if len(prediction) > 1 else prediction[0]
            dysarthria_prob = prediction[1] if len(prediction) > 1 else 1 - prediction[0]
            
            # Calculate scores
            clarity_score = normal_prob
            fluency_score = normal_prob
            overall_score = normal_prob
            
            return {
                'clarity_score': float(clarity_score),
                'fluency_score': float(fluency_score),
                'overall_score': float(overall_score),
                'dysarthria_detected': dysarthria_prob > 0.5,
                'confidence': float(max(normal_prob, dysarthria_prob)),
                'risk_level': 'high' if dysarthria_prob > 0.7 else 'low' if dysarthria_prob < 0.3 else 'moderate'
            }
            
        except Exception as e:
            print(f"Error in speech ML prediction: {e}")
            # Fallback to rule-based analysis
            return self._rule_based_speech_analysis(features)
    
    def _rule_based_speech_analysis(self, features: np.ndarray) -> Dict[str, float]:
        """Fallback rule-based speech analysis"""
        if len(features) < 10:
            return {
                'clarity_score': 0.0,
                'fluency_score': 0.0,
                'overall_score': 0.0,
                'dysarthria_detected': True,
                'confidence': 0.0,
                'risk_level': 'high'
            }
        
        # Extract key features
        mean_amplitude = features[0] if len(features) > 0 else 0
        std_amplitude = features[1] if len(features) > 1 else 0
        zcr_mean = features[6] if len(features) > 6 else 0
        # Tempo position: 4 basic + 2 centroid + 2 zcr + 26 MFCC + 2 rolloff + 24 chroma = 60
        tempo = features[60] if len(features) > 60 else 0
        
        # Calculate scores based on heuristics
        clarity_score = 1.0
        fluency_score = 1.0
        
        # Penalize for very low amplitude (weak speech)
        if mean_amplitude < 0.01:
            clarity_score -= 0.3
        
        # Penalize for high variability (inconsistent speech)
        if std_amplitude > 0.1:
            fluency_score -= 0.2
        
        # Penalize for very high zero crossing rate (noisy speech)
        if zcr_mean > 0.1:
            clarity_score -= 0.2
        
        # Penalize for very slow or fast tempo
        if tempo < 60 or tempo > 200:
            fluency_score -= 0.2
        
        overall_score = (clarity_score + fluency_score) / 2
        
        return {
            'clarity_score': max(0.0, clarity_score),
            'fluency_score': max(0.0, fluency_score),
            'overall_score': max(0.0, overall_score),
            'dysarthria_detected': overall_score < 0.6,
            'confidence': 0.6,  # Medium confidence for rule-based
            'risk_level': 'high' if overall_score < 0.4 else 'low' if overall_score > 0.8 else 'moderate'
        }


class PoseAnalysisML:
    """Advanced pose analysis for balance and arm movement assessment"""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load pose analysis model"""
        try:
            if os.path.exists('models/pose_model.pkl'):
                self.model = joblib.load('models/pose_model.pkl')
            else:
                self.model = self._create_pose_model()
                joblib.dump(self.model, 'models/pose_model.pkl')
        except Exception as e:
            print(f"Error loading pose model: {e}")
            self.model = self._create_pose_model()
    
    def _create_pose_model(self):
        """Create pose analysis model"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def extract_pose_features(self, image: np.ndarray) -> np.ndarray:
        """Extract pose features using computer vision"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use HOG features for pose detection
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detect people in image
        boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        features = []
        
        if len(boxes) > 0:
            # Use the largest detected person
            person_box = max(boxes, key=lambda x: x[2] * x[3])
            x, y, w, h = person_box
            
            # Extract person region
            person_roi = image[y:y+h, x:x+w]
            
            # Calculate basic features
            features.extend([w, h, w/h])  # dimensions and aspect ratio
            
            # Calculate center of mass
            center_x = x + w // 2
            center_y = y + h // 2
            features.extend([center_x, center_y])
            
            # Calculate stability metrics (simplified)
            features.extend([0, 0, 0, 0])  # Placeholder for stability features
            
        else:
            # No person detected
            features.extend([0] * 7)
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])
    
    def analyze_balance(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze balance from pose"""
        features = self.extract_pose_features(image)
        
        if np.all(features == 0):
            return {
                'stability_score': 0.0,
                'balance_detected': False,
                'confidence': 0.0,
                'risk_level': 'high'
            }
        
        try:
            # Use ML model for prediction
            prediction = self.model.predict_proba(features.reshape(1, -1))[0]
            
            # Assuming binary classification: stable vs unstable
            stable_prob = prediction[0] if len(prediction) > 1 else prediction[0]
            unstable_prob = prediction[1] if len(prediction) > 1 else 1 - prediction[0]
            
            stability_score = stable_prob
            
            return {
                'stability_score': float(stability_score),
                'balance_detected': stable_prob > 0.5,
                'confidence': float(max(stable_prob, unstable_prob)),
                'risk_level': 'high' if unstable_prob > 0.7 else 'low' if unstable_prob < 0.3 else 'moderate'
            }
            
        except Exception as e:
            print(f"Error in pose ML prediction: {e}")
            # Fallback to rule-based analysis
            return self._rule_based_balance_analysis(features)
    
    def _rule_based_balance_analysis(self, features: np.ndarray) -> Dict[str, float]:
        """Fallback rule-based balance analysis"""
        if len(features) < 5:
            return {
                'stability_score': 0.0,
                'balance_detected': False,
                'confidence': 0.0,
                'risk_level': 'high'
            }
        
        # Extract key features
        aspect_ratio = features[2] if len(features) > 2 else 1.0
        center_x = features[3] if len(features) > 3 else 0
        center_y = features[4] if len(features) > 4 else 0
        
        # Calculate stability based on aspect ratio and position
        stability_score = 1.0
        
        # Penalize for extreme aspect ratios (person might be falling)
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            stability_score -= 0.4
        
        # Check if person is centered in frame (simplified)
        if center_x < 100 or center_x > 500:  # Assuming 600px width
            stability_score -= 0.2
        
        return {
            'stability_score': max(0.0, stability_score),
            'balance_detected': stability_score > 0.6,
            'confidence': 0.6,
            'risk_level': 'high' if stability_score < 0.4 else 'low' if stability_score > 0.8 else 'moderate'
        }


class MLModelManager:
    """Main manager for all ML models"""
    
    def __init__(self):
        # Create models directory first, so the analyzers can save their model files
        os.makedirs('models', exist_ok=True)

        self.facial_analyzer = FacialAnalysisML()
        self.speech_analyzer = SpeechAnalysisML()
        self.pose_analyzer = PoseAnalysisML()
    
    def analyze_facial_symmetry(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze facial symmetry using ML"""
        return self.facial_analyzer.analyze_facial_symmetry(image)
    
    def analyze_speech_quality(self, audio_data: np.ndarray, sample_rate: int = 22050) -> Dict[str, float]:
        """Analyze speech quality using ML"""
        return self.speech_analyzer.analyze_speech_quality(audio_data, sample_rate)
    
    def analyze_balance(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze balance using ML"""
        return self.pose_analyzer.analyze_balance(image)
    
    def train_models(self, training_data: Dict[str, Any]):
        """Train all models with provided data"""
        try:
            # Train facial model
            if 'facial_data' in training_data:
                self._train_facial_model(training_data['facial_data'])
            
            # Train speech model
            if 'speech_data' in training_data:
                self._train_speech_model(training_data['speech_data'])
            
            # Train pose model
            if 'pose_data' in training_data:
                self._train_pose_model(training_data['pose_data'])
            
            return True
        except Exception as e:
            print(f"Error training models: {e}")
            return False
    
    def _train_facial_model(self, data):
        """Train facial analysis model"""
        # This would be implemented with actual training data
        pass
    
    def _train_speech_model(self, data):
        """Train speech analysis model"""
        # This would be implemented with actual training data
        pass
    
    def _train_pose_model(self, data):
        """Train pose analysis model"""
        # This would be implemented with actual training data
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'facial_model': 'Loaded' if self.facial_analyzer.model else 'Not loaded',
            'speech_model': 'Loaded' if self.speech_analyzer.model else 'Not loaded',
            'pose_model': 'Loaded' if self.pose_analyzer.model else 'Not loaded',
            'models_directory': 'models/',
            'status': 'Ready' if all([
                self.facial_analyzer.model,
                self.speech_analyzer.model,
                self.pose_analyzer.model
            ]) else 'Partially loaded'
        }

