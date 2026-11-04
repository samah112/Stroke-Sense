"""
Daily Interaction Hub
Health reminders, tips, progress tracking, and educational content for stroke prevention
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Optional, Any
import json
import random
from datetime import datetime, timedelta, time
import uuid


class ReminderType(Enum):
    MEDICATION = "medication"
    BLOOD_PRESSURE = "blood_pressure"
    BLOOD_SUGAR = "blood_sugar"
    EXERCISE = "exercise"
    APPOINTMENT = "appointment"
    ASSESSMENT = "assessment"
    HYDRATION = "hydration"
    SLEEP = "sleep"


class TipCategory(Enum):
    NUTRITION = "nutrition"
    EXERCISE = "exercise"
    LIFESTYLE = "lifestyle"
    MEDICATION = "medication"
    PREVENTION = "prevention"
    EMERGENCY = "emergency"


class ActivityType(Enum):
    WALKING = "walking"
    RUNNING = "running"
    CYCLING = "cycling"
    SWIMMING = "swimming"
    STRENGTH_TRAINING = "strength_training"
    YOGA = "yoga"
    OTHER = "other"


@dataclass
class HealthReminder:
    """Health reminder configuration"""
    reminder_id: str
    type: ReminderType
    title: str
    message: str
    time: time
    is_active: bool = True
    repeat_days: List[int] = None  # 0=Monday, 6=Sunday
    last_triggered: Optional[str] = None
    user_response: Optional[str] = None
    
    def __post_init__(self):
        if self.repeat_days is None:
            self.repeat_days = list(range(7))  # Daily by default


@dataclass
class HealthTip:
    """Health tip with educational content"""
    tip_id: str
    category: TipCategory
    title: str
    content: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    target_audience: str = "general"
    source: Optional[str] = None
    created_date: str = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()


@dataclass
class ActivityRecord:
    """Physical activity record"""
    activity_id: str
    type: ActivityType
    duration_minutes: int
    intensity: str  # low, moderate, high
    calories_burned: Optional[int] = None
    distance_km: Optional[float] = None
    steps: Optional[int] = None
    date: str = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.date is None:
            self.date = datetime.now().isoformat()


@dataclass
class VitalSigns:
    """Vital signs measurements"""
    measurement_id: str
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    heart_rate: Optional[int] = None
    blood_sugar: Optional[float] = None
    weight: Optional[float] = None
    temperature: Optional[float] = None
    date: str = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.date is None:
            self.date = datetime.now().isoformat()


@dataclass
class MedicationRecord:
    """Medication adherence record"""
    medication_id: str
    name: str
    dosage: str
    frequency: str
    taken: bool
    date: str = None
    time: str = None
    side_effects: Optional[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.date is None:
            self.date = datetime.now().isoformat()
        if self.time is None:
            self.time = datetime.now().strftime("%H:%M")


@dataclass
class ProgressSummary:
    """Weekly/monthly progress summary"""
    period_start: str
    period_end: str
    total_activities: int
    total_exercise_minutes: int
    avg_blood_pressure: Dict[str, float]
    medication_adherence: float
    assessment_frequency: int
    goals_achieved: List[str]
    improvements: List[str]
    recommendations: List[str]


class HealthReminderManager:
    """Manages health reminders and notifications"""
    
    def __init__(self):
        self.reminders: List[HealthReminder] = []
        self.default_reminders = self._create_default_reminders()
        self.reminders.extend(self.default_reminders)
    
    def _create_default_reminders(self) -> List[HealthReminder]:
        """Create default health reminders"""
        return [
            HealthReminder(
                reminder_id=str(uuid.uuid4()),
                type=ReminderType.MEDICATION,
                title="Take Your Medication",
                message="Time to take your prescribed medication. Don't forget to track any side effects.",
                time=time(8, 0),  # 8:00 AM
                repeat_days=[0, 1, 2, 3, 4, 5, 6]
            ),
            HealthReminder(
                reminder_id=str(uuid.uuid4()),
                type=ReminderType.BLOOD_PRESSURE,
                title="Check Blood Pressure",
                message="Please measure and record your blood pressure. Keep track of your readings.",
                time=time(9, 0),  # 9:00 AM
                repeat_days=[0, 1, 2, 3, 4, 5, 6]
            ),
            HealthReminder(
                reminder_id=str(uuid.uuid4()),
                type=ReminderType.EXERCISE,
                title="Daily Exercise",
                message="Time for your daily exercise! Even 30 minutes of walking can reduce stroke risk by 20%.",
                time=time(18, 0),  # 6:00 PM
                repeat_days=[0, 1, 2, 3, 4, 5, 6]
            ),
            HealthReminder(
                reminder_id=str(uuid.uuid4()),
                type=ReminderType.HYDRATION,
                title="Stay Hydrated",
                message="Drink water regularly throughout the day. Proper hydration improves blood circulation.",
                time=time(12, 0),  # 12:00 PM
                repeat_days=[0, 1, 2, 3, 4, 5, 6]
            )
        ]
    
    def add_reminder(self, reminder: HealthReminder) -> bool:
        """Add new health reminder"""
        try:
            self.reminders.append(reminder)
            return True
        except Exception as e:
            print(f"Error adding reminder: {e}")
            return False
    
    def get_active_reminders(self) -> List[HealthReminder]:
        """Get all active reminders"""
        return [r for r in self.reminders if r.is_active]
    
    def get_reminders_for_time(self, target_time: time) -> List[HealthReminder]:
        """Get reminders scheduled for specific time"""
        return [r for r in self.reminders if r.time == target_time and r.is_active]
    
    def mark_reminder_triggered(self, reminder_id: str, user_response: str = None) -> bool:
        """Mark reminder as triggered"""
        for reminder in self.reminders:
            if reminder.reminder_id == reminder_id:
                reminder.last_triggered = datetime.now().isoformat()
                reminder.user_response = user_response
                return True
        return False
    
    def disable_reminder(self, reminder_id: str) -> bool:
        """Disable specific reminder"""
        for reminder in self.reminders:
            if reminder.reminder_id == reminder_id:
                reminder.is_active = False
                return True
        return False


class HealthTipsManager:
    """Manages health tips and educational content"""
    
    def __init__(self):
        self.tips: List[HealthTip] = []
        self.default_tips = self._create_default_tips()
        self.tips.extend(self.default_tips)
    
    def _create_default_tips(self) -> List[HealthTip]:
        """Create default health tips"""
        return [
            HealthTip(
                tip_id=str(uuid.uuid4()),
                category=TipCategory.NUTRITION,
                title="Drink Enough Water",
                content="Drinking enough water improves blood circulation and helps maintain healthy blood pressure. Aim for 8 glasses (64 ounces) per day.",
                priority=1
            ),
            HealthTip(
                tip_id=str(uuid.uuid4()),
                category=TipCategory.EXERCISE,
                title="30 Minutes Daily Walking",
                content="30 minutes of walking daily reduces stroke risk by 20%. Start with short walks and gradually increase duration and intensity.",
                priority=1
            ),
            HealthTip(
                tip_id=str(uuid.uuid4()),
                category=TipCategory.LIFESTYLE,
                title="Manage Stress",
                content="Chronic stress increases stroke risk. Practice relaxation techniques like deep breathing, meditation, or yoga.",
                priority=2
            ),
            HealthTip(
                tip_id=str(uuid.uuid4()),
                category=TipCategory.NUTRITION,
                title="Reduce Sodium Intake",
                content="High sodium intake increases blood pressure. Limit processed foods and use herbs and spices for flavor instead of salt.",
                priority=1
            ),
            HealthTip(
                tip_id=str(uuid.uuid4()),
                category=TipCategory.PREVENTION,
                title="Know the BE FAST Signs",
                content="Remember BE FAST: Balance, Eyes, Face, Arms, Speech, Time. If you notice any of these symptoms, seek immediate medical attention.",
                priority=1
            ),
            HealthTip(
                tip_id=str(uuid.uuid4()),
                category=TipCategory.MEDICATION,
                title="Take Medications as Prescribed",
                content="Consistent medication adherence is crucial for stroke prevention. Set reminders and use pill organizers to stay on track.",
                priority=1
            ),
            HealthTip(
                tip_id=str(uuid.uuid4()),
                category=TipCategory.LIFESTYLE,
                title="Get Quality Sleep",
                content="Aim for 7-9 hours of quality sleep per night. Poor sleep increases stroke risk and affects overall health.",
                priority=2
            ),
            HealthTip(
                tip_id=str(uuid.uuid4()),
                category=TipCategory.NUTRITION,
                title="Eat More Fruits and Vegetables",
                content="A diet rich in fruits and vegetables provides antioxidants and nutrients that protect against stroke. Aim for 5 servings daily.",
                priority=2
            )
        ]
    
    def get_tips_by_category(self, category: TipCategory) -> List[HealthTip]:
        """Get tips by category"""
        return [t for t in self.tips if t.category == category]
    
    def get_random_tip(self, category: Optional[TipCategory] = None) -> Optional[HealthTip]:
        """Get random tip, optionally filtered by category"""
        filtered_tips = self.tips
        if category:
            filtered_tips = [t for t in self.tips if t.category == category]
        
        return random.choice(filtered_tips) if filtered_tips else None
    
    def get_daily_tip(self) -> HealthTip:
        """Get tip of the day based on date"""
        today = datetime.now().day
        tip_index = today % len(self.tips)
        return self.tips[tip_index]
    
    def add_tip(self, tip: HealthTip) -> bool:
        """Add new health tip"""
        try:
            self.tips.append(tip)
            return True
        except Exception as e:
            print(f"Error adding tip: {e}")
            return False


class ProgressTracker:
    """Tracks health progress and generates reports"""
    
    def __init__(self):
        self.activities: List[ActivityRecord] = []
        self.vital_signs: List[VitalSigns] = []
        self.medications: List[MedicationRecord] = []
        self.assessments: List[Dict] = []
    
    def add_activity(self, activity: ActivityRecord) -> bool:
        """Add activity record"""
        try:
            self.activities.append(activity)
            return True
        except Exception as e:
            print(f"Error adding activity: {e}")
            return False
    
    def add_vital_signs(self, vitals: VitalSigns) -> bool:
        """Add vital signs measurement"""
        try:
            self.vital_signs.append(vitals)
            return True
        except Exception as e:
            print(f"Error adding vital signs: {e}")
            return False
    
    def add_medication_record(self, medication: MedicationRecord) -> bool:
        """Add medication record"""
        try:
            self.medications.append(medication)
            return True
        except Exception as e:
            print(f"Error adding medication record: {e}")
            return False
    
    def add_assessment_record(self, assessment_data: Dict) -> bool:
        """Add assessment record"""
        try:
            assessment_data["timestamp"] = datetime.now().isoformat()
            self.assessments.append(assessment_data)
            return True
        except Exception as e:
            print(f"Error adding assessment record: {e}")
            return False
    
    def get_weekly_summary(self, week_start: datetime) -> ProgressSummary:
        """Generate weekly progress summary"""
        week_end = week_start + timedelta(days=7)
        
        # Filter data for the week
        week_activities = [a for a in self.activities 
                          if week_start <= datetime.fromisoformat(a.date) < week_end]
        week_vitals = [v for v in self.vital_signs 
                      if week_start <= datetime.fromisoformat(v.date) < week_end]
        week_medications = [m for m in self.medications 
                           if week_start <= datetime.fromisoformat(m.date) < week_end]
        week_assessments = [a for a in self.assessments 
                           if week_start <= datetime.fromisoformat(a["timestamp"]) < week_end]
        
        # Calculate metrics
        total_exercise_minutes = sum(a.duration_minutes for a in week_activities)
        
        # Calculate average blood pressure
        bp_readings = [(v.systolic_bp, v.diastolic_bp) for v in week_vitals 
                      if v.systolic_bp and v.diastolic_bp]
        avg_bp = {
            "systolic": sum(bp[0] for bp in bp_readings) / len(bp_readings) if bp_readings else 0,
            "diastolic": sum(bp[1] for bp in bp_readings) / len(bp_readings) if bp_readings else 0
        }
        
        # Calculate medication adherence
        total_medications = len(week_medications)
        taken_medications = len([m for m in week_medications if m.taken])
        adherence_rate = (taken_medications / total_medications * 100) if total_medications > 0 else 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(week_activities, week_vitals, week_medications)
        
        return ProgressSummary(
            period_start=week_start.isoformat(),
            period_end=week_end.isoformat(),
            total_activities=len(week_activities),
            total_exercise_minutes=total_exercise_minutes,
            avg_blood_pressure=avg_bp,
            medication_adherence=adherence_rate,
            assessment_frequency=len(week_assessments),
            goals_achieved=self._identify_goals_achieved(week_activities, week_vitals),
            improvements=self._identify_improvements(week_activities, week_vitals),
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, activities: List[ActivityRecord], 
                                 vitals: List[VitalSigns], 
                                 medications: List[MedicationRecord]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Exercise recommendations
        total_exercise = sum(a.duration_minutes for a in activities)
        if total_exercise < 150:  # Less than 2.5 hours per week
            recommendations.append("Increase physical activity to at least 150 minutes per week")
        
        # Blood pressure recommendations
        recent_bp = [v for v in vitals if v.systolic_bp and v.diastolic_bp]
        if recent_bp:
            avg_systolic = sum(v.systolic_bp for v in recent_bp) / len(recent_bp)
            if avg_systolic > 140:
                recommendations.append("Blood pressure is elevated - consult your doctor")
        
        # Medication adherence recommendations
        adherence_rate = len([m for m in medications if m.taken]) / len(medications) * 100 if medications else 0
        if adherence_rate < 80:
            recommendations.append("Improve medication adherence - consider setting reminders")
        
        return recommendations
    
    def _identify_goals_achieved(self, activities: List[ActivityRecord], 
                                vitals: List[VitalSigns]) -> List[str]:
        """Identify goals that were achieved"""
        goals = []
        
        # Exercise goals
        total_exercise = sum(a.duration_minutes for a in activities)
        if total_exercise >= 150:
            goals.append("Met weekly exercise goal (150+ minutes)")
        
        # Blood pressure goals
        recent_bp = [v for v in vitals if v.systolic_bp and v.diastolic_bp]
        if recent_bp:
            avg_systolic = sum(v.systolic_bp for v in recent_bp) / len(recent_bp)
            if avg_systolic < 120:
                goals.append("Maintained healthy blood pressure")
        
        return goals
    
    def _identify_improvements(self, activities: List[ActivityRecord], 
                              vitals: List[VitalSigns]) -> List[str]:
        """Identify areas of improvement"""
        improvements = []
        
        # Compare with previous week (simplified)
        if len(activities) > 0:
            improvements.append("Consistent physical activity recorded")
        
        if len(vitals) > 0:
            improvements.append("Regular vital signs monitoring")
        
        return improvements


class DailyHubManager:
    """Main manager for daily interaction hub"""
    
    def __init__(self):
        self.reminder_manager = HealthReminderManager()
        self.tips_manager = HealthTipsManager()
        self.progress_tracker = ProgressTracker()
        self.user_preferences = {}
    
    def get_daily_content(self) -> Dict[str, Any]:
        """Get daily content including tips, reminders, and progress"""
        return {
            "tip_of_the_day": self.tips_manager.get_daily_tip(),
            "active_reminders": self.reminder_manager.get_active_reminders(),
            "weekly_summary": self._get_current_week_summary(),
            "motivational_message": self._get_motivational_message(),
            "quick_actions": self._get_quick_actions()
        }
    
    def _get_current_week_summary(self) -> ProgressSummary:
        """Get current week's progress summary"""
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        return self.progress_tracker.get_weekly_summary(week_start)
    
    def _get_motivational_message(self) -> str:
        """Get motivational message based on progress"""
        messages = [
            "Every step towards better health counts! Keep up the great work!",
            "Your commitment to monitoring your health is inspiring. Stay consistent!",
            "Remember: prevention is better than cure. You're doing amazing!",
            "Small daily actions lead to big health improvements. Keep going!",
            "Your health journey is unique and valuable. Trust the process!"
        ]
        return random.choice(messages)
    
    def _get_quick_actions(self) -> List[Dict[str, str]]:
        """Get quick action buttons for the hub"""
        return [
            {
                "title": "Record Blood Pressure",
                "action": "record_bp",
                "icon": "heart"
            },
            {
                "title": "Log Exercise",
                "action": "log_exercise",
                "icon": "fitness"
            },
            {
                "title": "Take Assessment",
                "action": "start_assessment",
                "icon": "assessment"
            },
            {
                "title": "View Progress",
                "action": "view_progress",
                "icon": "chart"
            }
        ]
    
    def export_data(self, filepath: str) -> bool:
        """Export all hub data to JSON file"""
        try:
            export_data = {
                "reminders": [asdict(r) for r in self.reminder_manager.reminders],
                "tips": [asdict(t) for t in self.tips_manager.tips],
                "activities": [asdict(a) for a in self.progress_tracker.activities],
                "vital_signs": [asdict(v) for v in self.progress_tracker.vital_signs],
                "medications": [asdict(m) for m in self.progress_tracker.medications],
                "assessments": self.progress_tracker.assessments,
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=self._json_default)

            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Convert values that json cannot serialize (Enum, time, datetime)"""
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (time, datetime)):
            return value.isoformat()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
    
    def import_data(self, filepath: str) -> bool:
        """Import hub data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            # Import reminders
            if "reminders" in import_data:
                for reminder_data in import_data["reminders"]:
                    # Restore types that were converted to strings on export
                    reminder_data["type"] = ReminderType(reminder_data["type"])
                    reminder_data["time"] = time.fromisoformat(reminder_data["time"])
                    reminder = HealthReminder(**reminder_data)
                    self.reminder_manager.reminders.append(reminder)

            # Import tips
            if "tips" in import_data:
                for tip_data in import_data["tips"]:
                    tip_data["category"] = TipCategory(tip_data["category"])
                    tip = HealthTip(**tip_data)
                    self.tips_manager.tips.append(tip)
            
            # Import activities
            if "activities" in import_data:
                for activity_data in import_data["activities"]:
                    activity_data["type"] = ActivityType(activity_data["type"])
                    activity = ActivityRecord(**activity_data)
                    self.progress_tracker.activities.append(activity)
            
            # Import vital signs
            if "vital_signs" in import_data:
                for vitals_data in import_data["vital_signs"]:
                    vitals = VitalSigns(**vitals_data)
                    self.progress_tracker.vital_signs.append(vitals)
            
            # Import medications
            if "medications" in import_data:
                for medication_data in import_data["medications"]:
                    medication = MedicationRecord(**medication_data)
                    self.progress_tracker.medications.append(medication)
            
            # Import assessments
            if "assessments" in import_data:
                self.progress_tracker.assessments.extend(import_data["assessments"])
            
            return True
        except Exception as e:
            print(f"Error importing data: {e}")
            return False
