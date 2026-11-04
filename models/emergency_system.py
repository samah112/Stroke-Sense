"""
Emergency Response System
Handles automated emergency contacts, location tracking, and critical alert management
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Optional, Callable
import json
import requests
from datetime import datetime, timedelta
import geocoder
import time


class EmergencyContactType(Enum):
    EMERGENCY_SERVICES = "emergency_services"  # 911/Ambulance
    NEUROLOGIST = "neurologist"  # Medical specialist
    FAMILY_MEMBER = "family_member"  # First-degree relative
    CAREGIVER = "caregiver"  # Professional caregiver


class AlertPriority(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EmergencyContact:
    """Emergency contact information"""
    name: str
    phone_number: str
    contact_type: EmergencyContactType
    is_primary: bool = False
    is_active: bool = True
    relationship: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class LocationData:
    """Location information for emergency response"""
    latitude: float
    longitude: float
    address: Optional[str] = None
    accuracy: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class EmergencyAlert:
    """Emergency alert with all necessary information"""
    alert_id: str
    priority: AlertPriority
    risk_level: str
    symptoms_detected: List[str]
    location: LocationData
    timestamp: str
    user_id: str
    assessment_data: Dict
    contacts_notified: List[str] = None
    status: str = "active"
    
    def __post_init__(self):
        if self.contacts_notified is None:
            self.contacts_notified = []


class LocationService:
    """Handles location tracking and geocoding"""
    
    def __init__(self):
        self.last_location = None
        self.location_cache_duration = 300  # 5 minutes
    
    def get_current_location(self) -> Optional[LocationData]:
        """Get current location with caching"""
        current_time = datetime.now()
        
        # Check if we have a recent cached location
        if (self.last_location and 
            current_time - datetime.fromisoformat(self.last_location.timestamp) < 
            timedelta(seconds=self.location_cache_duration)):
            return self.last_location
        
        try:
            # Get location using geocoder
            g = geocoder.ip('me')
            if g.ok:
                location = LocationData(
                    latitude=g.lat,
                    longitude=g.lng,
                    address=g.address,
                    accuracy=g.accuracy
                )
                self.last_location = location
                return location
        except Exception as e:
            print(f"Error getting location: {e}")
        
        return None
    
    def get_address_from_coordinates(self, lat: float, lng: float) -> Optional[str]:
        """Get address from coordinates"""
        try:
            g = geocoder.reverse([lat, lng])
            if g.ok:
                return g.address
        except Exception as e:
            print(f"Error getting address: {e}")
        
        return None


class CommunicationService:
    """Handles emergency communications"""
    
    def __init__(self):
        self.sms_api_key = None  # Would be configured in real app
        self.email_api_key = None  # Would be configured in real app
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS message"""
        try:
            # In real implementation, would use Twilio or similar service
            print(f"SMS to {phone_number}: {message}")
            return True
        except Exception as e:
            print(f"Error sending SMS: {e}")
            return False
    
    def send_email(self, email: str, subject: str, message: str) -> bool:
        """Send email message"""
        try:
            # In real implementation, would use SendGrid or similar service
            print(f"Email to {email}: {subject} - {message}")
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def make_phone_call(self, phone_number: str, message: str) -> bool:
        """Make automated phone call"""
        try:
            # In real implementation, would use Twilio Voice API
            print(f"Calling {phone_number}: {message}")
            return True
        except Exception as e:
            print(f"Error making call: {e}")
            return False


class EmergencyResponseManager:
    """Main emergency response management system"""
    
    def __init__(self):
        self.contacts: List[EmergencyContact] = []
        self.location_service = LocationService()
        self.communication_service = CommunicationService()
        self.active_alerts: Dict[str, EmergencyAlert] = {}
        self.alert_history: List[EmergencyAlert] = []
        
        # Load default emergency contacts
        self._load_default_contacts()
    
    def _load_default_contacts(self):
        """Load default emergency contacts"""
        default_contacts = [
            EmergencyContact(
                name="Emergency Services",
                phone_number="911",
                contact_type=EmergencyContactType.EMERGENCY_SERVICES,
                is_primary=True
            )
        ]
        self.contacts.extend(default_contacts)
    
    def add_emergency_contact(self, contact: EmergencyContact) -> bool:
        """Add new emergency contact"""
        try:
            # Validate phone number format
            if not self._validate_phone_number(contact.phone_number):
                return False
            
            # Check for duplicates
            if any(c.phone_number == contact.phone_number for c in self.contacts):
                return False
            
            self.contacts.append(contact)
            return True
        except Exception as e:
            print(f"Error adding contact: {e}")
            return False
    
    def remove_emergency_contact(self, phone_number: str) -> bool:
        """Remove emergency contact"""
        try:
            self.contacts = [c for c in self.contacts if c.phone_number != phone_number]
            return True
        except Exception as e:
            print(f"Error removing contact: {e}")
            return False
    
    def update_emergency_contact(self, phone_number: str, updated_contact: EmergencyContact) -> bool:
        """Update existing emergency contact"""
        try:
            for i, contact in enumerate(self.contacts):
                if contact.phone_number == phone_number:
                    self.contacts[i] = updated_contact
                    return True
            return False
        except Exception as e:
            print(f"Error updating contact: {e}")
            return False
    
    def get_emergency_contacts(self) -> List[EmergencyContact]:
        """Get all emergency contacts"""
        return [c for c in self.contacts if c.is_active]
    
    def get_primary_contacts(self) -> List[EmergencyContact]:
        """Get primary emergency contacts"""
        return [c for c in self.contacts if c.is_primary and c.is_active]
    
    def trigger_emergency_alert(self, 
                               risk_level: str, 
                               symptoms: List[str], 
                               assessment_data: Dict,
                               user_id: str = "default_user") -> EmergencyAlert:
        """Trigger emergency alert and notify contacts"""
        
        # Get current location
        location = self.location_service.get_current_location()
        if not location:
            # Use default location if GPS fails
            location = LocationData(latitude=0.0, longitude=0.0, address="Location unavailable")
        
        # Create alert
        alert_id = f"alert_{int(time.time())}"
        priority = self._determine_priority(risk_level)
        
        alert = EmergencyAlert(
            alert_id=alert_id,
            priority=priority,
            risk_level=risk_level,
            symptoms_detected=symptoms,
            location=location,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            assessment_data=assessment_data
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Notify contacts based on priority
        self._notify_contacts(alert)
        
        return alert
    
    def _determine_priority(self, risk_level: str) -> AlertPriority:
        """Determine alert priority from risk level"""
        priority_mapping = {
            "Low": AlertPriority.LOW,
            "Moderate": AlertPriority.MODERATE,
            "High": AlertPriority.HIGH,
            "Critical": AlertPriority.CRITICAL,
            "Very High": AlertPriority.CRITICAL
        }
        
        return priority_mapping.get(risk_level, AlertPriority.MODERATE)
    
    def _notify_contacts(self, alert: EmergencyAlert):
        """Notify emergency contacts based on alert priority"""
        contacts_to_notify = self._get_contacts_for_priority(alert.priority)
        
        for contact in contacts_to_notify:
            success = self._send_alert_notification(contact, alert)
            if success:
                alert.contacts_notified.append(contact.phone_number)
    
    def _get_contacts_for_priority(self, priority: AlertPriority) -> List[EmergencyContact]:
        """Get contacts to notify based on priority"""
        if priority == AlertPriority.CRITICAL:
            # Notify all primary contacts
            return self.get_primary_contacts()
        elif priority == AlertPriority.HIGH:
            # Notify emergency services and neurologist
            return [c for c in self.contacts if c.contact_type in 
                   [EmergencyContactType.EMERGENCY_SERVICES, EmergencyContactType.NEUROLOGIST] and c.is_active]
        elif priority == AlertPriority.MODERATE:
            # Notify neurologist and family
            return [c for c in self.contacts if c.contact_type in 
                   [EmergencyContactType.NEUROLOGIST, EmergencyContactType.FAMILY_MEMBER] and c.is_active]
        else:  # LOW
            # Notify family only
            return [c for c in self.contacts if c.contact_type == EmergencyContactType.FAMILY_MEMBER and c.is_active]
    
    def _send_alert_notification(self, contact: EmergencyContact, alert: EmergencyAlert) -> bool:
        """Send notification to specific contact"""
        message = self._create_alert_message(alert, contact)
        
        if contact.contact_type == EmergencyContactType.EMERGENCY_SERVICES:
            # For emergency services, make phone call
            return self.communication_service.make_phone_call(contact.phone_number, message)
        else:
            # For other contacts, send SMS
            return self.communication_service.send_sms(contact.phone_number, message)
    
    def _create_alert_message(self, alert: EmergencyAlert, contact: EmergencyContact) -> str:
        """Create alert message for specific contact"""
        if contact.contact_type == EmergencyContactType.EMERGENCY_SERVICES:
            message = f"STROKE EMERGENCY ALERT - {alert.priority.value.upper()} PRIORITY\n"
            message += f"Time: {alert.timestamp}\n"
            message += f"Location: {alert.location.address or 'GPS coordinates available'}\n"
            message += f"Symptoms: {', '.join(alert.symptoms_detected)}\n"
            message += f"Risk Level: {alert.risk_level}\n"
            message += "Immediate medical response required."
        else:
            message = f"StrokeSense Alert - {alert.priority.value.upper()} Priority\n"
            message += f"Time: {alert.timestamp}\n"
            message += f"Location: {alert.location.address or 'Location data available'}\n"
            message += f"Symptoms detected: {', '.join(alert.symptoms_detected)}\n"
            message += f"Risk Level: {alert.risk_level}\n"
            message += "Please check on the user immediately."
        
        return message
    
    def _validate_phone_number(self, phone_number: str) -> bool:
        """Validate phone number format"""
        # Simple validation - in real app would use more sophisticated validation
        import re
        pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
        return bool(re.match(pattern, phone_number))
    
    def get_active_alerts(self) -> List[EmergencyAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 50) -> List[EmergencyAlert]:
        """Get alert history"""
        return self.alert_history[-limit:]
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> bool:
        """Mark alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = "resolved"
            alert.assessment_data["resolution_notes"] = resolution_notes
            alert.assessment_data["resolved_at"] = datetime.now().isoformat()
            
            # The alert is already in alert_history (added when it was triggered),
            # so only remove it from the active list - appending again duplicated it
            del self.active_alerts[alert_id]
            
            return True
        return False
    
    def get_emergency_statistics(self) -> Dict:
        """Get emergency response statistics"""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        resolved_alerts = total_alerts - active_alerts
        
        priority_counts = {}
        for alert in self.alert_history:
            priority = alert.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": resolved_alerts,
            "priority_distribution": priority_counts,
            "contacts_configured": len(self.get_emergency_contacts()),
            "last_alert": self.alert_history[-1].timestamp if self.alert_history else None
        }
    
    def export_alert_data(self, filepath: str) -> bool:
        """Export alert data to JSON file"""
        try:
            export_data = {
                "alerts": [asdict(alert) for alert in self.alert_history],
                "contacts": [asdict(contact) for contact in self.contacts],
                "statistics": self.get_emergency_statistics(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
    
    def import_alert_data(self, filepath: str) -> bool:
        """Import alert data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            # Import contacts
            if "contacts" in import_data:
                for contact_data in import_data["contacts"]:
                    contact = EmergencyContact(**contact_data)
                    self.contacts.append(contact)
            
            # Import alerts
            if "alerts" in import_data:
                for alert_data in import_data["alerts"]:
                    # Convert location data
                    if "location" in alert_data:
                        alert_data["location"] = LocationData(**alert_data["location"])
                    
                    alert = EmergencyAlert(**alert_data)
                    self.alert_history.append(alert)
            
            return True
        except Exception as e:
            print(f"Error importing data: {e}")
            return False


class EmergencyResponseConfig:
    """Configuration for emergency response system"""
    
    def __init__(self):
        self.auto_call_emergency = True
        self.auto_sms_family = True
        self.alert_timeout_seconds = 15
        self.location_update_interval = 60
        self.max_alert_history = 1000
        self.enable_gps_tracking = True
        self.enable_background_location = True
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            "auto_call_emergency": self.auto_call_emergency,
            "auto_sms_family": self.auto_sms_family,
            "alert_timeout_seconds": self.alert_timeout_seconds,
            "location_update_interval": self.location_update_interval,
            "max_alert_history": self.max_alert_history,
            "enable_gps_tracking": self.enable_gps_tracking,
            "enable_background_location": self.enable_background_location
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'EmergencyResponseConfig':
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
