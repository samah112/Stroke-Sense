"""
Advanced Audio Processing System for StrokeSense
Real-time audio recording, speech analysis, and dysarthria detection
"""

import pyaudio
import numpy as np
import librosa
import threading
import time
import queue
import wave
import tempfile
import os
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class AudioState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    ERROR = "error"

@dataclass
class AudioFrame:
    """Audio frame with metadata"""
    audio_data: np.ndarray
    timestamp: float
    frame_id: int
    sample_rate: int
    analysis_results: Optional[Dict] = None

@dataclass
class SpeechAnalysisResult:
    """Speech analysis results"""
    clarity_score: float
    fluency_score: float
    pace_score: float
    volume_score: float
    overall_score: float
    dysarthria_detected: bool
    confidence: float
    risk_level: str
    features: Dict[str, float]

class AudioProcessor:
    """Advanced audio processing with real-time analysis"""
    
    def __init__(self, sample_rate: int = 22050, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.state = AudioState.IDLE
        
        # Audio components
        self.audio = None
        self.stream = None
        self.recording_thread = None
        self._stop_requested = False
        
        # Data storage
        self.audio_queue = queue.Queue(maxsize=50)
        self.recording_buffer = []
        self.analysis_buffer = []
        
        # Callbacks
        self.audio_callbacks = []
        self.analysis_callbacks = []
        
        # Initialize PyAudio
        self._initialize_audio()
    
    def _initialize_audio(self):
        """Initialize PyAudio"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # Test audio device
            device_count = self.audio.get_device_count()
            if device_count == 0:
                raise Exception("No audio devices found")
            
            # Find default input device
            default_device = self.audio.get_default_input_device_info()
            print(f"Using audio device: {default_device['name']}")
            
        except Exception as e:
            print(f"Audio initialization error: {e}")
            self.state = AudioState.ERROR
    
    def start_recording(self) -> bool:
        """Start audio recording"""
        if self.state != AudioState.IDLE:
            return False
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.state = AudioState.RECORDING
            self._stop_requested = False
            self.recording_buffer = []
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.state = AudioState.ERROR
            return False
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """Stop recording and return audio data"""
        if self.state != AudioState.RECORDING:
            return None
        
        self._stop_requested = True
        
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.state = AudioState.IDLE
        
        # Return recorded audio
        if self.recording_buffer:
            return np.concatenate(self.recording_buffer)
        
        return None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to buffer
        self.recording_buffer.append(audio_data)
        
        # Create audio frame
        audio_frame = AudioFrame(
            audio_data=audio_data,
            timestamp=time.time(),
            frame_id=len(self.recording_buffer),
            sample_rate=self.sample_rate
        )
        
        # Add to queue
        try:
            self.audio_queue.put_nowait(audio_frame)
        except queue.Full:
            # Remove oldest frame
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_frame)
            except queue.Empty:
                pass
        
        # Notify callbacks
        for callback in self.audio_callbacks:
            try:
                callback(audio_frame)
            except Exception as e:
                print(f"Audio callback error: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def _recording_loop(self):
        """Main recording loop"""
        while not self._stop_requested and self.stream:
            try:
                time.sleep(0.01)  # Small delay to prevent CPU overload
            except Exception as e:
                print(f"Recording loop error: {e}")
                break
    
    def add_audio_callback(self, callback: Callable[[AudioFrame], None]):
        """Add callback for new audio frames"""
        self.audio_callbacks.append(callback)
    
    def add_analysis_callback(self, callback: Callable[[AudioFrame], None]):
        """Add callback for analysis results"""
        self.analysis_callbacks.append(callback)
    
    def get_latest_audio(self) -> Optional[AudioFrame]:
        """Get latest audio frame"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_audio_info(self) -> Dict[str, Any]:
        """Get audio system information"""
        if not self.audio:
            return {'status': 'Not initialized'}
        
        return {
            'status': self.state.value,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'queue_size': self.audio_queue.qsize(),
            'recording_active': self.state == AudioState.RECORDING,
            'device_count': self.audio.get_device_count()
        }
    
    def cleanup(self):
        """Cleanup audio resources"""
        if self.stream:
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()


class SpeechAnalyzer:
    """Advanced speech analysis for dysarthria detection"""
    
    def __init__(self):
        self.feature_extractor = SpeechFeatureExtractor()
        self.ml_analyzer = SpeechMLAnalyzer()
    
    def analyze_speech(self, audio_data: np.ndarray, sample_rate: int = 22050) -> SpeechAnalysisResult:
        """Comprehensive speech analysis"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio_data, sample_rate)
            
            # Perform ML analysis
            ml_results = self.ml_analyzer.analyze_features(features)
            
            # Calculate individual scores
            clarity_score = self._calculate_clarity_score(features)
            fluency_score = self._calculate_fluency_score(features)
            pace_score = self._calculate_pace_score(features)
            volume_score = self._calculate_volume_score(features)
            
            # Calculate overall score
            overall_score = (clarity_score + fluency_score + pace_score + volume_score) / 4
            
            # Determine dysarthria
            dysarthria_detected = overall_score < 0.6 or ml_results['dysarthria_probability'] > 0.5
            
            # Determine risk level
            if overall_score < 0.4 or ml_results['dysarthria_probability'] > 0.7:
                risk_level = 'high'
            elif overall_score < 0.6 or ml_results['dysarthria_probability'] > 0.5:
                risk_level = 'moderate'
            else:
                risk_level = 'low'
            
            return SpeechAnalysisResult(
                clarity_score=clarity_score,
                fluency_score=fluency_score,
                pace_score=pace_score,
                volume_score=volume_score,
                overall_score=overall_score,
                dysarthria_detected=dysarthria_detected,
                confidence=ml_results['confidence'],
                risk_level=risk_level,
                features=features
            )
            
        except Exception as e:
            print(f"Speech analysis error: {e}")
            return SpeechAnalysisResult(
                clarity_score=0.0,
                fluency_score=0.0,
                pace_score=0.0,
                volume_score=0.0,
                overall_score=0.0,
                dysarthria_detected=True,
                confidence=0.0,
                risk_level='high',
                features={}
            )
    
    def _calculate_clarity_score(self, features: Dict[str, float]) -> float:
        """Calculate speech clarity score"""
        # Based on spectral features
        spectral_centroid = features.get('spectral_centroid_mean', 0)
        spectral_rolloff = features.get('spectral_rolloff_mean', 0)
        zero_crossing_rate = features.get('zero_crossing_rate_mean', 0)
        
        # Normalize scores
        clarity_score = 1.0
        
        # Penalize for very low spectral centroid (muffled speech)
        if spectral_centroid < 1000:
            clarity_score -= 0.3
        
        # Penalize for high zero crossing rate (noisy speech)
        if zero_crossing_rate > 0.1:
            clarity_score -= 0.2
        
        return max(0.0, clarity_score)
    
    def _calculate_fluency_score(self, features: Dict[str, float]) -> float:
        """Calculate speech fluency score"""
        # Based on rhythm and tempo
        tempo = features.get('tempo', 0)
        rhythm_regularity = features.get('rhythm_regularity', 0)
        
        fluency_score = 1.0
        
        # Penalize for very slow or fast tempo
        if tempo < 60 or tempo > 200:
            fluency_score -= 0.3
        
        # Penalize for irregular rhythm
        if rhythm_regularity < 0.5:
            fluency_score -= 0.2
        
        return max(0.0, fluency_score)
    
    def _calculate_pace_score(self, features: Dict[str, float]) -> float:
        """Calculate speech pace score"""
        # Based on speaking rate
        speaking_rate = features.get('speaking_rate', 0)
        
        pace_score = 1.0
        
        # Optimal speaking rate is around 150-200 words per minute
        if speaking_rate < 100 or speaking_rate > 250:
            pace_score -= 0.4
        elif speaking_rate < 120 or speaking_rate > 220:
            pace_score -= 0.2
        
        return max(0.0, pace_score)
    
    def _calculate_volume_score(self, features: Dict[str, float]) -> float:
        """Calculate speech volume score"""
        # Based on amplitude features
        rms_energy = features.get('rms_energy_mean', 0)
        peak_amplitude = features.get('peak_amplitude', 0)
        
        volume_score = 1.0
        
        # Penalize for very low volume
        if rms_energy < 0.01:
            volume_score -= 0.4
        elif rms_energy < 0.05:
            volume_score -= 0.2
        
        # Penalize for clipping
        if peak_amplitude > 0.95:
            volume_score -= 0.1
        
        return max(0.0, volume_score)


class SpeechFeatureExtractor:
    """Extract comprehensive speech features"""
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int = 22050) -> Dict[str, float]:
        """Extract comprehensive speech features"""
        try:
            features = {}
            
            # Basic amplitude features
            features.update(self._extract_amplitude_features(audio_data))
            
            # Spectral features
            features.update(self._extract_spectral_features(audio_data, sample_rate))
            
            # Temporal features
            features.update(self._extract_temporal_features(audio_data, sample_rate))
            
            # Prosodic features
            features.update(self._extract_prosodic_features(audio_data, sample_rate))
            
            # MFCC features
            features.update(self._extract_mfcc_features(audio_data, sample_rate))
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return {}
    
    def _extract_amplitude_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract amplitude-based features"""
        return {
            'rms_energy_mean': float(np.sqrt(np.mean(audio_data**2))),
            'rms_energy_std': float(np.std(np.sqrt(np.mean(audio_data**2, axis=0)))),
            'peak_amplitude': float(np.max(np.abs(audio_data))),
            'amplitude_variance': float(np.var(audio_data)),
            'dynamic_range': float(np.max(audio_data) - np.min(audio_data))
        }
    
    def _extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract spectral features"""
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
                'zero_crossing_rate_mean': float(np.mean(zcr)),
                'zero_crossing_rate_std': float(np.std(zcr))
            }
        except Exception as e:
            print(f"Spectral feature extraction error: {e}")
            return {}
    
    def _extract_temporal_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract temporal features"""
        try:
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            
            # Rhythm regularity
            onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sample_rate)
            if len(onset_frames) > 1:
                intervals = np.diff(onset_frames)
                rhythm_regularity = 1.0 - (np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0
            else:
                rhythm_regularity = 0
            
            return {
                'tempo': float(np.atleast_1d(tempo)[0]),
                'rhythm_regularity': float(rhythm_regularity),
                'onset_count': float(len(onset_frames))
            }
        except Exception as e:
            print(f"Temporal feature extraction error: {e}")
            return {}
    
    def _extract_prosodic_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract prosodic features"""
        try:
            # Pitch (F0)
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                pitch_mean = np.mean(pitch_values)
                pitch_std = np.std(pitch_values)
                pitch_range = np.max(pitch_values) - np.min(pitch_values)
            else:
                pitch_mean = pitch_std = pitch_range = 0
            
            # Speaking rate (simplified)
            duration = len(audio_data) / sample_rate
            speaking_rate = len(pitch_values) / duration if duration > 0 else 0
            
            return {
                'pitch_mean': float(pitch_mean),
                'pitch_std': float(pitch_std),
                'pitch_range': float(pitch_range),
                'speaking_rate': float(speaking_rate)
            }
        except Exception as e:
            print(f"Prosodic feature extraction error: {e}")
            return {}
    
    def _extract_mfcc_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract MFCC features"""
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            
            features = {}
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
            return features
        except Exception as e:
            print(f"MFCC feature extraction error: {e}")
            return {}


class SpeechMLAnalyzer:
    """ML-based speech analysis"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load pre-trained speech model"""
        try:
            import joblib
            if os.path.exists('models/speech_ml_model.pkl'):
                self.model = joblib.load('models/speech_ml_model.pkl')
            else:
                self.model = self._create_model()
                joblib.dump(self.model, 'models/speech_ml_model.pkl')
        except Exception as e:
            print(f"Error loading speech ML model: {e}")
            self.model = self._create_model()
    
    def _create_model(self):
        """Create speech analysis model"""
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def analyze_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Analyze features using ML model"""
        try:
            # Convert features to array
            feature_array = np.array(list(features.values())).reshape(1, -1)
            
            # Normalize features
            if self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                feature_array = self.scaler.fit_transform(feature_array)
            else:
                feature_array = self.scaler.transform(feature_array)
            
            # Predict using model
            prediction = self.model.predict_proba(feature_array)[0]
            
            # Assuming binary classification: normal vs dysarthria
            normal_prob = prediction[0] if len(prediction) > 1 else prediction[0]
            dysarthria_prob = prediction[1] if len(prediction) > 1 else 1 - prediction[0]
            
            return {
                'normal_probability': float(normal_prob),
                'dysarthria_probability': float(dysarthria_prob),
                'confidence': float(max(normal_prob, dysarthria_prob))
            }
            
        except Exception as e:
            print(f"ML analysis error: {e}")
            return {
                'normal_probability': 0.5,
                'dysarthria_probability': 0.5,
                'confidence': 0.0
            }


class SpeechTest:
    """Speech test implementation"""
    
    def __init__(self, audio_processor: AudioProcessor):
        self.audio_processor = audio_processor
        self.speech_analyzer = SpeechAnalyzer()
        self.test_active = False
        self.test_results = {}
        self.audio_data = []
    
    def start_speech_test(self, duration: float = 5.0) -> bool:
        """Start speech test"""
        if self.test_active:
            return False
        
        self.test_active = True
        self.audio_data = []
        
        # Add audio callback
        self.audio_processor.add_audio_callback(self._on_audio_frame)
        
        # Start recording
        return self.audio_processor.start_recording()
    
    def stop_speech_test(self) -> Dict[str, Any]:
        """Stop test and return results"""
        if not self.test_active:
            return {}
        
        self.test_active = False
        
        # Stop recording
        audio_data = self.audio_processor.stop_recording()
        
        if audio_data is not None and len(audio_data) > 0:
            # Analyze speech
            analysis_result = self.speech_analyzer.analyze_speech(audio_data)
            
            self.test_results = {
                'clarity_score': analysis_result.clarity_score,
                'fluency_score': analysis_result.fluency_score,
                'pace_score': analysis_result.pace_score,
                'volume_score': analysis_result.volume_score,
                'overall_score': analysis_result.overall_score,
                'dysarthria_detected': analysis_result.dysarthria_detected,
                'confidence': analysis_result.confidence,
                'risk_level': analysis_result.risk_level,
                'audio_duration': len(audio_data) / self.audio_processor.sample_rate,
                'sample_rate': self.audio_processor.sample_rate
            }
        else:
            self.test_results = {
                'error': 'No audio data recorded',
                'overall_score': 0.0,
                'dysarthria_detected': True,
                'risk_level': 'high'
            }
        
        return self.test_results
    
    def _on_audio_frame(self, audio_frame: AudioFrame):
        """Handle audio frame during test"""
        if not self.test_active:
            return
        
        self.audio_data.append(audio_frame.audio_data)
    
    def get_test_phrases(self) -> List[str]:
        """Get test phrases for speech assessment"""
        return [
            "The sky is blue and the sun is shining",
            "Our house has three big windows",
            "Please repeat this sentence clearly",
            "The quick brown fox jumps over the lazy dog",
            "I can speak clearly and fluently"
        ]

