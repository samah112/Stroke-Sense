"""
Enhanced Firebase Backend Integration
Comprehensive data storage, ML model processing, and real-time synchronization
"""

import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
from firebase_admin.exceptions import FirebaseError
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import uuid
import base64
import io
from PIL import Image
import cv2


class FirebaseConfig:
    """Firebase configuration management"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.credentials = None
        self.app = None
        self.db = None
        self.bucket = None
        
    def initialize(self, service_account_path: str = None) -> bool:
        """Initialize Firebase services"""
        try:
            if service_account_path:
                cred = credentials.Certificate(service_account_path)
            else:
                # Use default credentials (for production)
                cred = credentials.ApplicationDefault()
            
            self.app = firebase_admin.initialize_app(cred, {
                'storageBucket': 'strokesense-app.appspot.com'  # Replace with your bucket
            })
            
            self.db = firestore.client()
            self.bucket = storage.bucket()
            self.credentials = cred
            
            return True
        except Exception as e:
            print(f"Firebase initialization error: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if Firebase is initialized"""
        return self.app is not None and self.db is not None


class UserDataManager:
    """Manages user data in Firestore"""
    
    def __init__(self, db):
        self.db = db
        self.users_collection = "users"
        self.assessments_collection = "assessments"
        self.emergency_alerts_collection = "emergency_alerts"
        self.progress_collection = "progress"
    
    def create_user_profile(self, user_id: str, profile_data: Dict) -> bool:
        """Create new user profile"""
        try:
            user_doc = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "profile": profile_data,
                "settings": {
                    "notifications_enabled": True,
                    "emergency_contacts_configured": False,
                    "baseline_model_created": False,
                    "assessment_frequency": "weekly"
                },
                "fsrp_assessment": None,
                "baseline_facial_model": None,
                "emergency_contacts": [],
                "health_goals": []
            }
            
            self.db.collection(self.users_collection).document(user_id).set(user_doc)
            return True
        except Exception as e:
            print(f"Error creating user profile: {e}")
            return False
    
    def update_user_profile(self, user_id: str, updates: Dict) -> bool:
        """Update user profile"""
        try:
            updates["last_updated"] = datetime.now().isoformat()
            self.db.collection(self.users_collection).document(user_id).update(updates)
            return True
        except Exception as e:
            print(f"Error updating user profile: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile"""
        try:
            doc = self.db.collection(self.users_collection).document(user_id).get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            print(f"Error getting user profile: {e}")
            return None
    
    def save_fsrp_assessment(self, user_id: str, assessment_data: Dict) -> bool:
        """Save FSRP assessment results"""
        try:
            assessment_doc = {
                "user_id": user_id,
                "assessment_type": "fsrp",
                "timestamp": datetime.now().isoformat(),
                "data": assessment_data,
                "risk_level": assessment_data.get("risk_level"),
                "score": assessment_data.get("score")
            }
            
            self.db.collection(self.assessments_collection).add(assessment_doc)
            
            # Update user profile with latest assessment
            self.update_user_profile(user_id, {
                "fsrp_assessment": assessment_data,
                "last_fsrp_assessment": datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            print(f"Error saving FSRP assessment: {e}")
            return False
    
    def save_befast_assessment(self, user_id: str, assessment_data: Dict) -> bool:
        """Save BE FAST assessment results"""
        try:
            assessment_doc = {
                "user_id": user_id,
                "assessment_type": "befast",
                "timestamp": datetime.now().isoformat(),
                "data": assessment_data,
                "overall_score": assessment_data.get("overall_score"),
                "risk_level": assessment_data.get("risk_level"),
                "components": assessment_data.get("components", {})
            }
            
            self.db.collection(self.assessments_collection).add(assessment_doc)
            
            # Update user profile with latest assessment
            self.update_user_profile(user_id, {
                "last_befast_assessment": datetime.now().isoformat(),
                "last_assessment_score": assessment_data.get("overall_score")
            })
            
            return True
        except Exception as e:
            print(f"Error saving BE FAST assessment: {e}")
            return False
    
    def save_emergency_alert(self, alert_data: Dict) -> bool:
        """Save emergency alert"""
        try:
            alert_doc = {
                "alert_id": alert_data.get("alert_id"),
                "user_id": alert_data.get("user_id"),
                "timestamp": datetime.now().isoformat(),
                "priority": alert_data.get("priority"),
                "risk_level": alert_data.get("risk_level"),
                "symptoms": alert_data.get("symptoms", []),
                "location": alert_data.get("location", {}),
                "status": alert_data.get("status", "active"),
                "contacts_notified": alert_data.get("contacts_notified", []),
                "assessment_data": alert_data.get("assessment_data", {})
            }
            
            self.db.collection(self.emergency_alerts_collection).add(alert_doc)
            return True
        except Exception as e:
            print(f"Error saving emergency alert: {e}")
            return False
    
    def save_progress_data(self, user_id: str, progress_type: str, data: Dict) -> bool:
        """Save progress tracking data"""
        try:
            progress_doc = {
                "user_id": user_id,
                "progress_type": progress_type,  # activity, vital_signs, medication, etc.
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            self.db.collection(self.progress_collection).add(progress_doc)
            return True
        except Exception as e:
            print(f"Error saving progress data: {e}")
            return False
    
    def get_user_assessments(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's assessment history"""
        try:
            query = (self.db.collection(self.assessments_collection)
                    .where("user_id", "==", user_id)
                    .order_by("timestamp", direction=firestore.Query.DESCENDING)
                    .limit(limit))
            
            assessments = []
            for doc in query.stream():
                assessments.append(doc.to_dict())
            
            return assessments
        except Exception as e:
            print(f"Error getting user assessments: {e}")
            return []
    
    def get_emergency_alerts(self, user_id: str = None, limit: int = 100) -> List[Dict]:
        """Get emergency alerts"""
        try:
            query = self.db.collection(self.emergency_alerts_collection)
            
            if user_id:
                query = query.where("user_id", "==", user_id)
            
            query = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
            
            alerts = []
            for doc in query.stream():
                alerts.append(doc.to_dict())
            
            return alerts
        except Exception as e:
            print(f"Error getting emergency alerts: {e}")
            return []


class MediaStorageManager:
    """Manages media file storage in Firebase Storage"""
    
    def __init__(self, bucket):
        self.bucket = bucket
        self.audio_folder = "audio"
        self.image_folder = "images"
        self.video_folder = "videos"
        self.baseline_folder = "baseline_models"
    
    def upload_image(self, image_data: np.ndarray, user_id: str, filename: str = None) -> Optional[str]:
        """Upload image to Firebase Storage"""
        try:
            if filename is None:
                filename = f"{user_id}_{int(datetime.now().timestamp())}.jpg"
            
            # Convert numpy array to PIL Image
            if len(image_data.shape) == 3:
                image = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image_data)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=85)
            img_buffer.seek(0)
            
            # Upload to Firebase Storage
            blob_path = f"{self.image_folder}/{filename}"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_file(img_buffer, content_type='image/jpeg')
            
            # Make blob publicly accessible (optional)
            blob.make_public()
            
            return blob.public_url
        except Exception as e:
            print(f"Error uploading image: {e}")
            return None
    
    def upload_audio(self, audio_data: bytes, user_id: str, filename: str = None) -> Optional[str]:
        """Upload audio file to Firebase Storage"""
        try:
            if filename is None:
                filename = f"{user_id}_{int(datetime.now().timestamp())}.wav"
            
            blob_path = f"{self.audio_folder}/{filename}"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(audio_data, content_type='audio/wav')
            
            return blob.public_url
        except Exception as e:
            print(f"Error uploading audio: {e}")
            return None
    
    def upload_video(self, video_data: bytes, user_id: str, filename: str = None) -> Optional[str]:
        """Upload video file to Firebase Storage"""
        try:
            if filename is None:
                filename = f"{user_id}_{int(datetime.now().timestamp())}.mp4"
            
            blob_path = f"{self.video_folder}/{filename}"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(video_data, content_type='video/mp4')
            
            return blob.public_url
        except Exception as e:
            print(f"Error uploading video: {e}")
            return None
    
    def upload_baseline_model(self, model_data: Dict, user_id: str) -> Optional[str]:
        """Upload baseline facial model"""
        try:
            filename = f"{user_id}_baseline_model.json"
            blob_path = f"{self.baseline_folder}/{filename}"
            
            # Convert model data to JSON
            json_data = json.dumps(model_data, indent=2)
            
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(json_data, content_type='application/json')
            
            return blob.public_url
        except Exception as e:
            print(f"Error uploading baseline model: {e}")
            return None
    
    def download_file(self, blob_path: str) -> Optional[bytes]:
        """Download file from Firebase Storage"""
        try:
            blob = self.bucket.blob(blob_path)
            return blob.download_as_bytes()
        except Exception as e:
            print(f"Error downloading file: {e}")
            return None
    
    def delete_file(self, blob_path: str) -> bool:
        """Delete file from Firebase Storage"""
        try:
            blob = self.bucket.blob(blob_path)
            blob.delete()
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False


class MLModelProcessor:
    """Handles ML model processing and inference"""
    
    def __init__(self, storage_manager: MediaStorageManager):
        self.storage_manager = storage_manager
        self.models = {}
        self.model_endpoints = {
            "facial_analysis": "https://your-ml-endpoint.com/facial-analysis",
            "speech_analysis": "https://your-ml-endpoint.com/speech-analysis",
            "pose_analysis": "https://your-ml-endpoint.com/pose-analysis"
        }
    
    def process_facial_analysis(self, image_url: str, user_id: str) -> Dict:
        """Process facial analysis using ML model"""
        try:
            # Download image from storage
            image_data = self.storage_manager.download_file(image_url)
            if not image_data:
                return {"error": "Failed to download image"}
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            # Process with ML model (simplified - would use actual ML service)
            analysis_result = self._run_facial_ml_model(image_array)
            
            # Save results to database
            result_data = {
                "user_id": user_id,
                "image_url": image_url,
                "analysis_result": analysis_result,
                "timestamp": datetime.now().isoformat(),
                "model_version": "1.0"
            }
            
            return result_data
        except Exception as e:
            print(f"Error processing facial analysis: {e}")
            return {"error": str(e)}
    
    def process_speech_analysis(self, audio_url: str, user_id: str) -> Dict:
        """Process speech analysis using ML model"""
        try:
            # Download audio from storage
            audio_data = self.storage_manager.download_file(audio_url)
            if not audio_data:
                return {"error": "Failed to download audio"}
            
            # Process with ML model (simplified - would use actual ML service)
            analysis_result = self._run_speech_ml_model(audio_data)
            
            # Save results to database
            result_data = {
                "user_id": user_id,
                "audio_url": audio_url,
                "analysis_result": analysis_result,
                "timestamp": datetime.now().isoformat(),
                "model_version": "1.0"
            }
            
            return result_data
        except Exception as e:
            print(f"Error processing speech analysis: {e}")
            return {"error": str(e)}
    
    def _run_facial_ml_model(self, image_array: np.ndarray) -> Dict:
        """Run facial analysis ML model (placeholder)"""
        # In real implementation, this would call your trained ML model
        return {
            "symmetry_score": 0.85,
            "facial_landmarks": [],
            "expression_analysis": "neutral",
            "asymmetry_detected": False,
            "confidence": 0.92
        }
    
    def _run_speech_ml_model(self, audio_data: bytes) -> Dict:
        """Run speech analysis ML model (placeholder)"""
        # In real implementation, this would call your trained ML model
        return {
            "clarity_score": 0.88,
            "fluency_score": 0.82,
            "dysarthria_detected": False,
            "speech_features": {},
            "confidence": 0.90
        }


class RealTimeSyncManager:
    """Manages real-time data synchronization"""
    
    def __init__(self, db):
        self.db = db
        self.listeners = {}
    
    def start_listening_to_user_data(self, user_id: str, callback: callable) -> str:
        """Start listening to real-time user data changes"""
        try:
            def on_snapshot(doc_snapshot, changes, read_time):
                for change in changes:
                    if change.type.name == 'MODIFIED':
                        callback(change.document.to_dict())
            
            listener = self.db.collection("users").document(user_id).on_snapshot(on_snapshot)
            listener_id = str(uuid.uuid4())
            self.listeners[listener_id] = listener
            
            return listener_id
        except Exception as e:
            print(f"Error starting listener: {e}")
            return None
    
    def start_listening_to_emergency_alerts(self, user_id: str, callback: callable) -> str:
        """Start listening to emergency alerts"""
        try:
            def on_snapshot(doc_snapshot, changes, read_time):
                for change in changes:
                    if change.type.name == 'ADDED':
                        callback(change.document.to_dict())
            
            query = self.db.collection("emergency_alerts").where("user_id", "==", user_id)
            listener = query.on_snapshot(on_snapshot)
            listener_id = str(uuid.uuid4())
            self.listeners[listener_id] = listener
            
            return listener_id
        except Exception as e:
            print(f"Error starting emergency listener: {e}")
            return None
    
    def stop_listener(self, listener_id: str) -> bool:
        """Stop specific listener"""
        try:
            if listener_id in self.listeners:
                self.listeners[listener_id].unsubscribe()
                del self.listeners[listener_id]
                return True
            return False
        except Exception as e:
            print(f"Error stopping listener: {e}")
            return False
    
    def stop_all_listeners(self) -> bool:
        """Stop all active listeners"""
        try:
            for listener in self.listeners.values():
                listener.unsubscribe()
            self.listeners.clear()
            return True
        except Exception as e:
            print(f"Error stopping all listeners: {e}")
            return False


class StrokeSenseBackend:
    """Main backend manager for StrokeSense application"""
    
    def __init__(self, service_account_path: str = None):
        self.config = FirebaseConfig()
        self.user_manager = None
        self.storage_manager = None
        self.ml_processor = None
        self.sync_manager = None
        self.initialized = False
        
        if self.config.initialize(service_account_path):
            self.user_manager = UserDataManager(self.config.db)
            self.storage_manager = MediaStorageManager(self.config.bucket)
            self.ml_processor = MLModelProcessor(self.storage_manager)
            self.sync_manager = RealTimeSyncManager(self.config.db)
            self.initialized = True
    
    def is_ready(self) -> bool:
        """Check if backend is ready"""
        return self.initialized and self.config.is_initialized()
    
    def create_user(self, user_id: str, profile_data: Dict) -> bool:
        """Create new user"""
        if not self.is_ready():
            return False
        return self.user_manager.create_user_profile(user_id, profile_data)
    
    def save_assessment(self, user_id: str, assessment_type: str, data: Dict) -> bool:
        """Save assessment data"""
        if not self.is_ready():
            return False
        
        if assessment_type == "fsrp":
            return self.user_manager.save_fsrp_assessment(user_id, data)
        elif assessment_type == "befast":
            return self.user_manager.save_befast_assessment(user_id, data)
        else:
            return False
    
    def upload_media(self, media_type: str, data: Union[np.ndarray, bytes], user_id: str) -> Optional[str]:
        """Upload media file"""
        if not self.is_ready():
            return None
        
        if media_type == "image":
            return self.storage_manager.upload_image(data, user_id)
        elif media_type == "audio":
            return self.storage_manager.upload_audio(data, user_id)
        elif media_type == "video":
            return self.storage_manager.upload_video(data, user_id)
        else:
            return None
    
    def process_ml_analysis(self, analysis_type: str, media_url: str, user_id: str) -> Dict:
        """Process ML analysis"""
        if not self.is_ready():
            return {"error": "Backend not ready"}
        
        if analysis_type == "facial":
            return self.ml_processor.process_facial_analysis(media_url, user_id)
        elif analysis_type == "speech":
            return self.ml_processor.process_speech_analysis(media_url, user_id)
        else:
            return {"error": "Unknown analysis type"}
    
    def get_user_data(self, user_id: str) -> Optional[Dict]:
        """Get user data"""
        if not self.is_ready():
            return None
        return self.user_manager.get_user_profile(user_id)
    
    def get_assessment_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get assessment history"""
        if not self.is_ready():
            return []
        return self.user_manager.get_user_assessments(user_id, limit)
    
    def setup_real_time_sync(self, user_id: str, data_callback: callable, alert_callback: callable) -> Dict[str, str]:
        """Setup real-time synchronization"""
        if not self.is_ready():
            return {}
        
        listeners = {}
        
        data_listener = self.sync_manager.start_listening_to_user_data(user_id, data_callback)
        if data_listener:
            listeners["data"] = data_listener
        
        alert_listener = self.sync_manager.start_listening_to_emergency_alerts(user_id, alert_callback)
        if alert_listener:
            listeners["alerts"] = alert_listener
        
        return listeners
    
    def cleanup(self):
        """Cleanup resources"""
        if self.sync_manager:
            self.sync_manager.stop_all_listeners()
        
        if self.config.app:
            firebase_admin.delete_app(self.config.app)
