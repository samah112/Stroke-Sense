# StrokeSense 🧠

**AI-Powered Stroke Risk Assessment & Early Detection**

StrokeSense is a Python desktop application that combines clinical risk scoring with computer vision, speech analysis, and machine learning to help users assess their stroke risk, recognize early warning signs using the **BE FAST** method, and get help quickly in an emergency.

> ⚠️ **Medical disclaimer:** StrokeSense is an academic / research prototype. It is **not** a certified medical device and must not be used for diagnosis. If you suspect a stroke, call your local emergency number immediately.

---

## ✨ Features

| Module | Description |
|---|---|
| **FSRP Risk Assessment** | Questionnaire based on the Framingham Stroke Risk Profile that calculates a risk score, a risk level (Low → Very High) and personalized recommendations. |
| **BE FAST Assessment** | Guided test covering **B**alance, **E**yes, **F**ace, **A**rms, **S**peech and **T**ime. |
| **Facial Analysis** | Captures a baseline facial model and compares later images to detect facial asymmetry. |
| **Speech Analysis** | Records audio and extracts features (MFCC, spectral, pitch, tempo) with `librosa` to screen for dysarthria (slurred speech). |
| **Camera Processing** | Real-time pose estimation and eye tracking with `MediaPipe` and `OpenCV`. |
| **ML Models** | CNN (TensorFlow/Keras) for facial symmetry, plus Random Forest / SVM models for speech and balance analysis. |
| **Emergency System** | Emergency contacts, location lookup, and priority-based alerts. |
| **Daily Health Hub** | Medication / blood-pressure reminders, health tips, and weekly progress tracking. |
| **Firebase Backend** | Firestore and Cloud Storage integration for users, assessments, and media. |

---

## 📁 Project Structure

```
Stroke-Sense/
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── models/
│   ├── fsrp_model.py           # Framingham Stroke Risk Profile
│   ├── befast_assessment.py    # BE FAST assessment engine
│   ├── facial_analysis.py      # Facial landmarks, symmetry & baseline model
│   ├── ml_models.py            # Facial, speech & pose ML models
│   ├── emergency_system.py     # Emergency contacts & alerts
│   ├── daily_hub.py            # Reminders, tips & progress tracking
│   └── firebase_backend.py     # Firebase Firestore / Storage integration
├── processing/
│   ├── camera_processor.py     # Real-time camera, pose & eye tracking
│   └── audio_processor.py      # Audio recording & speech analysis
└── ui/
    └── strokesense_app.py      # CustomTkinter user interface
```

---

## 🚀 Getting Started

### Prerequisites
- Python **3.10**
- A webcam and a microphone (for the BE FAST tests)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/samah112/Stroke-Sense.git
cd Stroke-Sense

# 2. Create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** `pyaudio` needs PortAudio. On macOS run `brew install portaudio`; on Windows, install it with `pip install pyaudio` (pre-built wheels are available).

### Run the app

```bash
python main.py
```

### Firebase (optional)
To enable cloud storage, create a Firebase project, download a **service account key**, and pass its path to `StrokeSenseBackend(service_account_path=...)`.
Update `storageBucket` in `models/firebase_backend.py` to match your bucket.
**Never commit your service account key.** It is already excluded by `.gitignore`.

🔒 **Privacy:** face images, voice recordings and facial models are health data, so they are stored **privately**. Uploads return a storage path, not a public link. When a link is needed, use `get_temporary_url()`, which creates a signed URL that expires after 15 minutes.

---

## 🔄 How It Works

1. **Onboarding:** the user completes the FSRP questionnaire and receives a 10-year stroke risk estimate.
2. **Baseline setup:** a neutral facial image is captured as the user's personal baseline.
3. **BE FAST assessment:** the app guides the user through each test, and every component is scored from 0 to 1.
4. **Risk evaluation:** the scores are combined into an overall risk level (Low, Moderate, High or Critical).
5. **Emergency response:** a High or Critical result triggers an alert to the configured emergency contacts.

---

## 🛠️ Tech Stack

`Python` · `OpenCV` · `MediaPipe` · `TensorFlow / Keras` · `scikit-learn` · `librosa` · `PyAudio` · `CustomTkinter` · `Firebase Admin SDK`

---

## 📌 Project Status

This is a working prototype. Some components still use **simulated results or placeholders** and are planned for future work:
- ML models are created untrained by default; they need training on a clinical dataset.
- Some BE FAST steps in the UI (eyes, arms, speech, balance) currently return simulated scores.
- SMS, email and calls in the emergency system are mocked (they print to the console). They are designed to plug into a provider such as Twilio.
- The ML endpoints in `firebase_backend.py` are placeholders.
- The FSRP module uses a **simplified point system** inspired by the Framingham Stroke Risk Profile (score 0–21), not the official sex-specific FSRP tables. Risk percentages are approximate and capped at the top of the scale.
