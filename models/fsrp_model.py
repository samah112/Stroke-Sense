"""
Framingham Stroke Risk Profile (FSRP) Model
Implements the clinical evaluation tool for 10-year stroke risk assessment
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import json


class RiskLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"


class Gender(Enum):
    MALE = "Male"
    FEMALE = "Female"


class HypertensionTreatment(Enum):
    NO = "No"
    YES = "Yes"


class DiabetesStatus(Enum):
    NO = "No"
    YES = "Yes"


class SmokingStatus(Enum):
    NON_SMOKER = "Non-smoker"
    CURRENT_SMOKER = "Current smoker"


class HeartDiseaseStatus(Enum):
    NONE = "None"
    PRESENT = "Present (MI, CAD, CHF)"


class AtrialFibrillationStatus(Enum):
    NO = "No"
    YES = "Yes"


@dataclass
class FSRPAssessment:
    """Framingham Stroke Risk Profile Assessment"""
    age: int
    gender: Gender
    systolic_bp: int
    hypertension_treatment: HypertensionTreatment
    diabetes: DiabetesStatus
    smoking: SmokingStatus
    heart_disease: HeartDiseaseStatus
    atrial_fibrillation: AtrialFibrillationStatus
    
    def calculate_score(self) -> int:
        """Calculate FSRP risk score based on clinical guidelines"""
        score = 0
        
        # Age scoring
        if self.age < 55:
            score += 0
        elif 55 <= self.age <= 64:
            score += 2
        elif 65 <= self.age <= 74:
            score += 4
        else:  # age >= 75
            score += 6
        
        # Gender scoring
        if self.gender == Gender.FEMALE:
            score += 1
        
        # Systolic Blood Pressure scoring
        if self.systolic_bp < 120:
            score += 0
        elif 120 <= self.systolic_bp <= 129:
            score += 1
        elif 130 <= self.systolic_bp <= 139:
            score += 2
        else:  # >= 140
            score += 3
        
        # Hypertension Treatment scoring
        if self.hypertension_treatment == HypertensionTreatment.YES:
            score += 1
        
        # Diabetes scoring
        if self.diabetes == DiabetesStatus.YES:
            score += 2
        
        # Smoking scoring
        if self.smoking == SmokingStatus.CURRENT_SMOKER:
            score += 2
        
        # Heart Disease scoring
        if self.heart_disease == HeartDiseaseStatus.PRESENT:
            score += 2
        
        # Atrial Fibrillation scoring
        if self.atrial_fibrillation == AtrialFibrillationStatus.YES:
            score += 4
        
        return score
    
    def get_risk_level(self) -> RiskLevel:
        """Determine risk level based on calculated score"""
        score = self.calculate_score()
        
        if score <= 3:
            return RiskLevel.LOW
        elif score <= 6:
            return RiskLevel.MODERATE
        elif score <= 10:
            return RiskLevel.HIGH
        else:  # score >= 11
            return RiskLevel.VERY_HIGH
    
    def get_risk_percentage(self) -> float:
        """Estimate 10-year stroke risk percentage based on score"""
        score = self.calculate_score()
        
        # Approximate risk percentages based on FSRP guidelines
        risk_mapping = {
            0: 0.5, 1: 0.8, 2: 1.2, 3: 1.8,
            4: 2.7, 5: 4.0, 6: 5.9, 7: 8.7,
            8: 12.8, 9: 18.9, 10: 27.8, 11: 40.9,
            12: 60.2, 13: 88.5, 14: 130.2, 15: 191.5
        }
        
        return risk_mapping.get(min(score, 15), 200.0)
    
    def to_dict(self) -> Dict:
        """Convert assessment to dictionary for storage"""
        return {
            'age': self.age,
            'gender': self.gender.value,
            'systolic_bp': self.systolic_bp,
            'hypertension_treatment': self.hypertension_treatment.value,
            'diabetes': self.diabetes.value,
            'smoking': self.smoking.value,
            'heart_disease': self.heart_disease.value,
            'atrial_fibrillation': self.atrial_fibrillation.value,
            'score': self.calculate_score(),
            'risk_level': self.get_risk_level().value,
            'risk_percentage': self.get_risk_percentage()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FSRPAssessment':
        """Create assessment from dictionary"""
        return cls(
            age=data['age'],
            gender=Gender(data['gender']),
            systolic_bp=data['systolic_bp'],
            hypertension_treatment=HypertensionTreatment(data['hypertension_treatment']),
            diabetes=DiabetesStatus(data['diabetes']),
            smoking=SmokingStatus(data['smoking']),
            heart_disease=HeartDiseaseStatus(data['heart_disease']),
            atrial_fibrillation=AtrialFibrillationStatus(data['atrial_fibrillation'])
        )


class FSRPQuestionnaire:
    """Interactive questionnaire for FSRP assessment"""
    
    def __init__(self):
        self.questions = [
            {
                'id': 'age',
                'question': 'What is your age?',
                'type': 'number',
                'min': 18,
                'max': 120,
                'required': True
            },
            {
                'id': 'gender',
                'question': 'What is your gender?',
                'type': 'choice',
                'options': [Gender.MALE.value, Gender.FEMALE.value],
                'required': True
            },
            {
                'id': 'systolic_bp',
                'question': 'What is your systolic blood pressure (mmHg)?',
                'type': 'number',
                'min': 80,
                'max': 250,
                'required': True,
                'help_text': 'The top number when measuring blood pressure'
            },
            {
                'id': 'hypertension_treatment',
                'question': 'Are you currently taking medication for high blood pressure?',
                'type': 'choice',
                'options': [HypertensionTreatment.NO.value, HypertensionTreatment.YES.value],
                'required': True
            },
            {
                'id': 'diabetes',
                'question': 'Do you have diabetes?',
                'type': 'choice',
                'options': [DiabetesStatus.NO.value, DiabetesStatus.YES.value],
                'required': True
            },
            {
                'id': 'smoking',
                'question': 'What is your smoking status?',
                'type': 'choice',
                'options': [SmokingStatus.NON_SMOKER.value, SmokingStatus.CURRENT_SMOKER.value],
                'required': True
            },
            {
                'id': 'heart_disease',
                'question': 'Do you have any heart disease (heart attack, coronary artery disease, or heart failure)?',
                'type': 'choice',
                'options': [HeartDiseaseStatus.NONE.value, HeartDiseaseStatus.PRESENT.value],
                'required': True
            },
            {
                'id': 'atrial_fibrillation',
                'question': 'Do you have atrial fibrillation (irregular heartbeat)?',
                'type': 'choice',
                'options': [AtrialFibrillationStatus.NO.value, AtrialFibrillationStatus.YES.value],
                'required': True
            }
        ]
    
    def get_questions(self) -> List[Dict]:
        """Get all questionnaire questions"""
        return self.questions
    
    def validate_response(self, question_id: str, response: any) -> bool:
        """Validate user response for a specific question"""
        question = next((q for q in self.questions if q['id'] == question_id), None)
        if not question:
            return False
        
        if question['type'] == 'number':
            try:
                value = int(response)
                return question['min'] <= value <= question['max']
            except (ValueError, TypeError):
                return False
        
        elif question['type'] == 'choice':
            return response in question['options']
        
        return False
    
    def create_assessment(self, responses: Dict) -> Optional[FSRPAssessment]:
        """Create FSRP assessment from questionnaire responses"""
        try:
            # Validate all responses
            for question in self.questions:
                if question['required'] and question['id'] not in responses:
                    return None
                
                if question['id'] in responses:
                    if not self.validate_response(question['id'], responses[question['id']]):
                        return None
            
            # Create assessment
            return FSRPAssessment(
                age=int(responses['age']),
                gender=Gender(responses['gender']),
                systolic_bp=int(responses['systolic_bp']),
                hypertension_treatment=HypertensionTreatment(responses['hypertension_treatment']),
                diabetes=DiabetesStatus(responses['diabetes']),
                smoking=SmokingStatus(responses['smoking']),
                heart_disease=HeartDiseaseStatus(responses['heart_disease']),
                atrial_fibrillation=AtrialFibrillationStatus(responses['atrial_fibrillation'])
            )
        except (ValueError, KeyError):
            return None


def get_risk_recommendations(risk_level: RiskLevel) -> List[str]:
    """Get personalized recommendations based on risk level"""
    recommendations = {
        RiskLevel.LOW: [
            "Continue maintaining a healthy lifestyle",
            "Regular exercise and balanced diet",
            "Annual health checkups recommended",
            "Monitor blood pressure regularly"
        ],
        RiskLevel.MODERATE: [
            "Consider lifestyle modifications",
            "Regular exercise (30 minutes daily)",
            "Blood pressure monitoring every 6 months",
            "Discuss preventive measures with your doctor"
        ],
        RiskLevel.HIGH: [
            "Immediate lifestyle changes recommended",
            "Regular medical follow-ups",
            "Blood pressure monitoring monthly",
            "Consider medication consultation",
            "Stroke prevention education recommended"
        ],
        RiskLevel.VERY_HIGH: [
            "Immediate medical evaluation recommended",
            "Close medical supervision required",
            "Comprehensive stroke prevention plan",
            "Regular specialist consultations",
            "Emergency action plan preparation"
        ]
    }
    
    return recommendations.get(risk_level, [])
