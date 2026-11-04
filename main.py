"""
StrokeSense - AI-Powered Stroke Detection Mobile Application
Main application file integrating all components
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox, ttk
import customtkinter as ctk
import cv2
import numpy as np
import threading
import time
from datetime import datetime, timedelta
import json
import uuid
from typing import Dict, List, Optional, Any

# Import our custom modules
from models.fsrp_model import FSRPQuestionnaire, FSRPAssessment, RiskLevel
from models.facial_analysis import FacialAnalyzer, BaselineModel
from models.befast_assessment import BEFASTAssessmentEngine, BEFASTAssessment
from models.emergency_system import EmergencyResponseManager, EmergencyContact, EmergencyContactType
from models.daily_hub import DailyHubManager, HealthReminder, ReminderType
from models.firebase_backend import StrokeSenseBackend
from ui.strokesense_app import StrokeSenseApp, ColorScheme, Fonts


class StrokeSenseMainApp(StrokeSenseApp):
    """Enhanced StrokeSense application with full functionality"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize core components
        self.fsrp_questionnaire = FSRPQuestionnaire()
        self.facial_analyzer = FacialAnalyzer()
        self.baseline_model = BaselineModel()
        self.befast_engine = BEFASTAssessmentEngine()
        self.emergency_manager = EmergencyResponseManager()
        self.daily_hub = DailyHubManager()
        self.backend = StrokeSenseBackend()
        
        # User state
        self.current_user_id = str(uuid.uuid4())
        self.user_profile = None
        self.fsrp_completed = False
        self.baseline_created = False
        self.assessment_in_progress = False
        
        # Camera and media
        self.camera = None
        self.camera_active = False
        
        # Initialize user onboarding
        self._start_user_onboarding()
    
    def _new_onboarding_screen(self):
        """Create a fresh full-size screen for onboarding steps.

        The onboarding screens used to destroy every widget inside content_frame,
        which also destroyed the main views (dashboard, assessment, progress,
        emergency) and crashed the app when returning to the dashboard.
        Now the views are only hidden, and onboarding is drawn in its own frame.
        """
        self._close_onboarding_screen()
        for view in self.views.values():
            view.grid_remove()
        self.onboarding_screen = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.onboarding_screen.grid(row=0, column=0, sticky="nsew")
        return self.onboarding_screen

    def _close_onboarding_screen(self):
        """Remove the onboarding screen if one is shown"""
        screen = getattr(self, "onboarding_screen", None)
        if screen is not None:
            screen.destroy()
            self.onboarding_screen = None

    def _show_view(self, view_name: str):
        """Show a main view (override: close any onboarding screen first)"""
        self._close_onboarding_screen()
        super()._show_view(view_name)

    def _start_user_onboarding(self):
        """Start user onboarding process"""
        self._show_onboarding_welcome()
    
    def _show_onboarding_welcome(self):
        """Show welcome screen for new users"""
        # Show a fresh onboarding screen (keeps the main views intact)
        screen = self._new_onboarding_screen()
        
        # Create welcome screen
        welcome_frame = ctk.CTkFrame(screen)
        welcome_frame.pack(expand=True, fill="both", padx=50, pady=50)
        
        # Welcome title
        title_label = ctk.CTkLabel(
            welcome_frame,
            text="Welcome to StrokeSense",
            font=Fonts.HEADING_LARGE,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.pack(pady=(50, 20))
        
        # Welcome message
        message_label = ctk.CTkLabel(
            welcome_frame,
            text="Your AI-powered companion for stroke detection and prevention.\n\nLet's get started with a quick health assessment to personalize your experience.",
            font=Fonts.BODY_LARGE,
            text_color=ColorScheme.TEXT_SECONDARY,
            justify="center"
        )
        message_label.pack(pady=(0, 40))
        
        # Start button
        start_btn = ctk.CTkButton(
            welcome_frame,
            text="Start Health Assessment",
            command=self._start_fsrp_assessment,
            height=60,
            font=Fonts.BUTTON_LARGE,
            fg_color=ColorScheme.PRIMARY_BLUE,
            hover_color=self._darken_color(ColorScheme.PRIMARY_BLUE)
        )
        start_btn.pack(pady=(0, 50))
    
    def _start_fsrp_assessment(self):
        """Start FSRP health assessment"""
        self._show_fsrp_questionnaire()
    
    def _show_fsrp_questionnaire(self):
        """Show FSRP questionnaire"""
        # Show a fresh onboarding screen (keeps the main views intact)
        screen = self._new_onboarding_screen()
        
        # Create questionnaire frame
        questionnaire_frame = ctk.CTkFrame(screen)
        questionnaire_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            questionnaire_frame,
            text="Health Assessment (FSRP)",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.pack(pady=(20, 30))
        
        # Create scrollable frame for questions
        canvas = ctk.CTkCanvas(questionnaire_frame)
        scrollbar = ctk.CTkScrollbar(questionnaire_frame, orientation="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=20)
        scrollbar.pack(side="right", fill="y", padx=(0, 20), pady=20)
        
        # Store responses
        self.fsrp_responses = {}
        
        # Create question widgets
        questions = self.fsrp_questionnaire.get_questions()
        self.question_widgets = {}
        
        for i, question in enumerate(questions):
            question_frame = ctk.CTkFrame(scrollable_frame)
            question_frame.pack(fill="x", padx=20, pady=10)
            
            # Question text
            question_label = ctk.CTkLabel(
                question_frame,
                text=question['question'],
                font=Fonts.BODY_MEDIUM,
                text_color=ColorScheme.TEXT_PRIMARY,
                wraplength=600
            )
            question_label.pack(anchor="w", padx=20, pady=(20, 10))
            
            # Input widget based on type
            if question['type'] == 'number':
                entry = ctk.CTkEntry(
                    question_frame,
                    placeholder_text=f"Enter {question.get('help_text', 'value')}",
                    width=300,
                    height=40
                )
                entry.pack(anchor="w", padx=20, pady=(0, 20))
                self.question_widgets[question['id']] = entry
                
            elif question['type'] == 'choice':
                var = tk.StringVar()
                for option in question['options']:
                    radio = ctk.CTkRadioButton(
                        question_frame,
                        text=option,
                        variable=var,
                        value=option,
                        font=Fonts.BODY_MEDIUM
                    )
                    radio.pack(anchor="w", padx=40, pady=5)
                
                self.question_widgets[question['id']] = var
        
        # Submit button
        submit_frame = ctk.CTkFrame(scrollable_frame)
        submit_frame.pack(fill="x", padx=20, pady=30)
        
        submit_btn = ctk.CTkButton(
            submit_frame,
            text="Complete Assessment",
            command=self._submit_fsrp_assessment,
            height=50,
            font=Fonts.BUTTON_MEDIUM,
            fg_color=ColorScheme.SECONDARY_GREEN
        )
        submit_btn.pack(pady=20)
    
    def _submit_fsrp_assessment(self):
        """Submit FSRP assessment"""
        # Collect responses
        responses = {}
        for question_id, widget in self.question_widgets.items():
            if isinstance(widget, ctk.CTkEntry):
                value = widget.get()
                if value:
                    responses[question_id] = value
            elif isinstance(widget, tk.StringVar):
                value = widget.get()
                if value:
                    responses[question_id] = value
        
        # Validate responses
        if len(responses) != len(self.fsrp_questionnaire.get_questions()):
            messagebox.showerror("Incomplete Assessment", "Please answer all questions before submitting.")
            return
        
        # Create assessment
        assessment = self.fsrp_questionnaire.create_assessment(responses)
        if not assessment:
            messagebox.showerror("Invalid Responses", "Please check your answers and try again.")
            return
        
        # Store assessment
        self.user_profile = {
            "user_id": self.current_user_id,
            "fsrp_assessment": assessment.to_dict(),
            "created_at": datetime.now().isoformat()
        }
        
        # Save to backend
        if self.backend.is_ready():
            self.backend.save_assessment(self.current_user_id, "fsrp", assessment.to_dict())
        
        # Show results
        self._show_fsrp_results(assessment)
    
    def _show_fsrp_results(self, assessment: FSRPAssessment):
        """Show FSRP assessment results"""
        # Show a fresh onboarding screen (keeps the main views intact)
        screen = self._new_onboarding_screen()
        
        # Create results frame
        results_frame = ctk.CTkFrame(screen)
        results_frame.pack(expand=True, fill="both", padx=50, pady=50)
        
        # Title
        title_label = ctk.CTkLabel(
            results_frame,
            text="Health Assessment Results",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.pack(pady=(30, 20))
        
        # Risk level
        risk_level = assessment.get_risk_level()
        risk_color = self._get_risk_color(risk_level)
        
        risk_label = ctk.CTkLabel(
            results_frame,
            text=f"Your Risk Level: {risk_level.value}",
            font=Fonts.HEADING_SMALL,
            text_color=risk_color
        )
        risk_label.pack(pady=(0, 10))
        
        # Score
        score_label = ctk.CTkLabel(
            results_frame,
            text=f"Risk Score: {assessment.calculate_score()}/{assessment.MAX_SCORE}",
            font=Fonts.BODY_LARGE,
            text_color=ColorScheme.TEXT_PRIMARY
        )
        score_label.pack(pady=(0, 20))
        
        # Risk percentage
        percentage_label = ctk.CTkLabel(
            results_frame,
            text=f"10-Year Stroke Risk: {assessment.get_risk_percentage():.1f}%",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_SECONDARY
        )
        percentage_label.pack(pady=(0, 30))
        
        # Recommendations
        recommendations = self._get_risk_recommendations(risk_level)
        rec_label = ctk.CTkLabel(
            results_frame,
            text="Recommendations:",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_PRIMARY
        )
        rec_label.pack(anchor="w", padx=50, pady=(0, 10))
        
        for rec in recommendations:
            rec_item = ctk.CTkLabel(
                results_frame,
                text=f"• {rec}",
                font=Fonts.BODY_SMALL,
                text_color=ColorScheme.TEXT_SECONDARY
            )
            rec_item.pack(anchor="w", padx=70, pady=2)
        
        # Continue button
        continue_btn = ctk.CTkButton(
            results_frame,
            text="Continue to Baseline Setup",
            command=self._start_baseline_setup,
            height=50,
            font=Fonts.BUTTON_MEDIUM,
            fg_color=ColorScheme.PRIMARY_BLUE
        )
        continue_btn.pack(pady=(40, 30))
    
    def _get_risk_color(self, risk_level: RiskLevel) -> str:
        """Get color for risk level"""
        color_map = {
            RiskLevel.LOW: ColorScheme.SUCCESS,
            RiskLevel.MODERATE: ColorScheme.WARNING,
            RiskLevel.HIGH: ColorScheme.DANGER,
            RiskLevel.VERY_HIGH: ColorScheme.SECONDARY_RED
        }
        return color_map.get(risk_level, ColorScheme.TEXT_PRIMARY)
    
    def _get_risk_recommendations(self, risk_level: RiskLevel) -> List[str]:
        """Get recommendations for risk level"""
        recommendations = {
            RiskLevel.LOW: [
                "Continue maintaining a healthy lifestyle",
                "Regular exercise and balanced diet",
                "Annual health checkups recommended"
            ],
            RiskLevel.MODERATE: [
                "Consider lifestyle modifications",
                "Regular exercise (30 minutes daily)",
                "Blood pressure monitoring every 6 months"
            ],
            RiskLevel.HIGH: [
                "Immediate lifestyle changes recommended",
                "Regular medical follow-ups",
                "Blood pressure monitoring monthly"
            ],
            RiskLevel.VERY_HIGH: [
                "Immediate medical evaluation recommended",
                "Close medical supervision required",
                "Comprehensive stroke prevention plan"
            ]
        }
        return recommendations.get(risk_level, [])
    
    def _start_baseline_setup(self):
        """Start baseline facial model setup"""
        self._show_baseline_instructions()
    
    def _show_baseline_instructions(self):
        """Show baseline setup instructions"""
        # Show a fresh onboarding screen (keeps the main views intact)
        screen = self._new_onboarding_screen()
        
        # Create instructions frame
        instructions_frame = ctk.CTkFrame(screen)
        instructions_frame.pack(expand=True, fill="both", padx=50, pady=50)
        
        # Title
        title_label = ctk.CTkLabel(
            instructions_frame,
            text="Create Your Baseline Facial Model",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.pack(pady=(30, 20))
        
        # Instructions
        instructions_text = """
To create your baseline facial model, we need to capture a clear image of your face in a neutral expression.

Instructions:
• Sit in good lighting
• Look directly at the camera
• Keep a neutral facial expression
• Ensure your entire face is visible
• Hold still for a few seconds

This baseline will be used to detect any changes in your facial symmetry during future assessments.
        """
        
        instructions_label = ctk.CTkLabel(
            instructions_frame,
            text=instructions_text,
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_PRIMARY,
            justify="left"
        )
        instructions_label.pack(pady=(0, 30))
        
        # Start camera button
        start_camera_btn = ctk.CTkButton(
            instructions_frame,
            text="Start Camera",
            command=self._start_camera_for_baseline,
            height=50,
            font=Fonts.BUTTON_MEDIUM,
            fg_color=ColorScheme.PRIMARY_BLUE
        )
        start_camera_btn.pack(pady=(0, 30))
    
    def _start_camera_for_baseline(self):
        """Start camera for baseline capture"""
        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Camera Error", "Could not access camera. Please check your camera permissions.")
                return
            
            self.camera_active = True
            self._show_camera_capture()
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Error starting camera: {str(e)}")
    
    def _show_camera_capture(self):
        """Show camera capture interface"""
        # Show a fresh onboarding screen (keeps the main views intact)
        screen = self._new_onboarding_screen()
        
        # Create camera frame
        camera_frame = ctk.CTkFrame(screen)
        camera_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            camera_frame,
            text="Capture Baseline Image",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.pack(pady=(20, 10))
        
        # Camera preview area
        self.camera_preview = ctk.CTkLabel(
            camera_frame,
            text="Camera Preview",
            width=400,
            height=300,
            fg_color=ColorScheme.LIGHT_GRAY
        )
        self.camera_preview.pack(pady=20)
        
        # Control buttons
        button_frame = ctk.CTkFrame(camera_frame, fg_color="transparent")
        button_frame.pack(pady=20)
        
        capture_btn = ctk.CTkButton(
            button_frame,
            text="Capture Image",
            command=self._capture_baseline_image,
            height=50,
            font=Fonts.BUTTON_MEDIUM,
            fg_color=ColorScheme.SECONDARY_GREEN
        )
        capture_btn.pack(side="left", padx=10)
        
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self._cancel_camera,
            height=50,
            font=Fonts.BUTTON_MEDIUM,
            fg_color=ColorScheme.MEDIUM_GRAY
        )
        cancel_btn.pack(side="left", padx=10)
        
        # Start camera preview
        self._update_camera_preview()
    
    def _update_camera_preview(self):
        """Update camera preview"""
        if not self.camera_active or not self.camera:
            return
        
        try:
            ret, frame = self.camera.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for preview
                frame_resized = cv2.resize(frame_rgb, (400, 300))
                
                # Convert to PhotoImage
                from PIL import Image, ImageTk
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)
                
                # Update preview
                self.camera_preview.configure(image=photo)
                self.camera_preview.image = photo  # Keep a reference
                
                # Schedule next update
                self.after(30, self._update_camera_preview)
            
        except Exception as e:
            print(f"Camera preview error: {e}")
    
    def _capture_baseline_image(self):
        """Capture baseline image"""
        if not self.camera:
            return
        
        try:
            ret, frame = self.camera.read()
            if ret:
                # Analyze facial features
                features = self.facial_analyzer.analyze_facial_features(frame)
                
                if features and features.overall_symmetry > 0.7:
                    # Create baseline model
                    success = self.baseline_model.create_baseline(frame)
                    
                    if success:
                        # Save baseline
                        baseline_path = f"baseline_{self.current_user_id}.json"
                        self.baseline_model.save_baseline(baseline_path)
                        
                        # Update user profile
                        self.user_profile["baseline_model"] = baseline_path
                        self.user_profile["baseline_created"] = True
                        
                        # Stop camera
                        self._stop_camera()
                        
                        # Show success and continue
                        messagebox.showinfo("Success", "Baseline facial model created successfully!")
                        self._complete_onboarding()
                    else:
                        messagebox.showerror("Error", "Could not create baseline model. Please try again.")
                else:
                    messagebox.showerror("Poor Quality", "Please ensure your face is clearly visible and well-lit.")
            
        except Exception as e:
            messagebox.showerror("Capture Error", f"Error capturing image: {str(e)}")
    
    def _cancel_camera(self):
        """Cancel camera capture"""
        self._stop_camera()
        self._show_baseline_instructions()
    
    def _stop_camera(self):
        """Stop camera"""
        self.camera_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def _complete_onboarding(self):
        """Complete user onboarding"""
        self.fsrp_completed = True
        self.baseline_created = True
        
        # Update user status
        self.user_status_label.configure(text=f"User: {self.current_user_id[:8]}...")
        
        # Show main dashboard
        self._show_view("dashboard")
        
        # Show completion message
        messagebox.showinfo("Setup Complete", "Welcome to StrokeSense! Your health profile has been created and you're ready to start monitoring your health.")
    
    def _start_befast_assessment(self):
        """Start BE FAST assessment (override from parent)"""
        if not self.baseline_created:
            messagebox.showwarning("Setup Required", "Please complete the baseline setup first.")
            return
        
        if self.assessment_in_progress:
            messagebox.showwarning("Assessment in Progress", "An assessment is already in progress.")
            return
        
        self.assessment_in_progress = True
        self._show_befast_assessment()
    
    def _show_befast_assessment(self):
        """Show BE FAST assessment interface"""
        # Clear assessment area
        for widget in self.assessment_area.winfo_children():
            widget.destroy()
        
        # Create assessment steps
        steps_frame = ctk.CTkFrame(self.assessment_area)
        steps_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Assessment title
        title_label = ctk.CTkLabel(
            steps_frame,
            text="BE FAST Stroke Assessment",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.pack(pady=(20, 30))
        
        # Instructions
        instructions_label = ctk.CTkLabel(
            steps_frame,
            text="This assessment will guide you through the BE FAST evaluation.\nFollow the instructions for each component carefully.",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_SECONDARY,
            justify="center"
        )
        instructions_label.pack(pady=(0, 30))
        
        # Start assessment button
        start_btn = ctk.CTkButton(
            steps_frame,
            text="Start BE FAST Assessment",
            command=self._begin_befast_steps,
            height=60,
            font=Fonts.BUTTON_LARGE,
            fg_color=ColorScheme.PRIMARY_BLUE
        )
        start_btn.pack(pady=(0, 30))
    
    def _begin_befast_steps(self):
        """Begin BE FAST assessment steps"""
        self.befast_step = 0
        self.befast_results = {}
        self._show_befast_step()
    
    def _show_befast_step(self):
        """Show current BE FAST step"""
        steps = [
            ("Balance", "Stand with feet together and hold for 10 seconds", "balance"),
            ("Eyes", "Follow the moving dot with your eyes", "eyes"),
            ("Face", "Smile and show your teeth", "face"),
            ("Arms", "Raise both arms and hold for 10 seconds", "arms"),
            ("Speech", "Repeat: 'The sky is blue and the sun is shining'", "speech"),
            ("Time", "Assessment completed", "time")
        ]
        
        if self.befast_step >= len(steps):
            self._complete_befast_assessment()
            return
        
        step_name, instruction, step_type = steps[self.befast_step]
        
        # Clear previous content
        for widget in self.assessment_area.winfo_children():
            widget.destroy()
        
        # Create step frame
        step_frame = ctk.CTkFrame(self.assessment_area)
        step_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Step title
        title_label = ctk.CTkLabel(
            step_frame,
            text=f"Step {self.befast_step + 1}: {step_name}",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.pack(pady=(30, 20))
        
        # Instruction
        instruction_label = ctk.CTkLabel(
            step_frame,
            text=instruction,
            font=Fonts.BODY_LARGE,
            text_color=ColorScheme.TEXT_PRIMARY,
            wraplength=600
        )
        instruction_label.pack(pady=(0, 30))
        
        # Step-specific interface
        if step_type == "eyes":
            self._show_eye_tracking_interface(step_frame)
        elif step_type == "face":
            self._show_facial_expression_interface(step_frame)
        elif step_type == "arms":
            self._show_arm_movement_interface(step_frame)
        elif step_type == "speech":
            self._show_speech_interface(step_frame)
        else:
            self._show_generic_step_interface(step_frame, step_type)
    
    def _show_eye_tracking_interface(self, parent_frame):
        """Show eye tracking interface"""
        # Eye tracking instructions
        instructions_label = ctk.CTkLabel(
            parent_frame,
            text="A moving dot will appear on screen. Follow it with your eyes only.\nKeep your head still.",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_SECONDARY
        )
        instructions_label.pack(pady=(0, 20))
        
        # Eye tracking area
        self.eye_tracking_area = ctk.CTkFrame(parent_frame, width=400, height=300)
        self.eye_tracking_area.pack(pady=20)
        
        # Start eye tracking button
        start_btn = ctk.CTkButton(
            parent_frame,
            text="Start Eye Tracking",
            command=self._start_eye_tracking,
            height=50,
            font=Fonts.BUTTON_MEDIUM,
            fg_color=ColorScheme.SECONDARY_GREEN
        )
        start_btn.pack(pady=(0, 20))
    
    def _start_eye_tracking(self):
        """Start eye tracking test"""
        # Simulate eye tracking (in real app would use actual eye tracking)
        self.eye_tracking_area.configure(fg_color=ColorScheme.BLACK)
        
        # Create moving dot
        dot_label = ctk.CTkLabel(
            self.eye_tracking_area,
            text="●",
            font=("Arial", 20),
            text_color=ColorScheme.WHITE
        )
        dot_label.pack(expand=True)
        
        # Simulate movement
        self._animate_eye_tracking_dot(dot_label, 0)
    
    def _animate_eye_tracking_dot(self, dot_label, step):
        """Animate eye tracking dot"""
        if step < 10:  # 10 steps of movement
            # Move dot to different positions
            positions = [(100, 100), (300, 100), (300, 200), (100, 200), (200, 150)]
            pos = positions[step % len(positions)]
            
            dot_label.place(x=pos[0], y=pos[1])
            
            # Schedule next movement
            self.after(1000, lambda: self._animate_eye_tracking_dot(dot_label, step + 1))
        else:
            # Eye tracking complete
            self.befast_results["eyes"] = {"score": 0.8, "result": "Normal"}
            self._next_befast_step()
    
    def _show_facial_expression_interface(self, parent_frame):
        """Show facial expression interface"""
        # Instructions
        instructions_label = ctk.CTkLabel(
            parent_frame,
            text="Look at the camera and perform the requested facial expression.\nThe AI will analyze your facial symmetry.",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_SECONDARY
        )
        instructions_label.pack(pady=(0, 20))
        
        # Expression instruction
        self.expression_label = ctk.CTkLabel(
            parent_frame,
            text="Smile and show your teeth",
            font=Fonts.BODY_LARGE,
            text_color=ColorScheme.PRIMARY_BLUE
        )
        self.expression_label.pack(pady=(0, 20))
        
        # Camera preview for facial analysis
        self.facial_preview = ctk.CTkLabel(
            parent_frame,
            text="Camera Preview",
            width=300,
            height=200,
            fg_color=ColorScheme.LIGHT_GRAY
        )
        self.facial_preview.pack(pady=20)
        
        # Capture button
        capture_btn = ctk.CTkButton(
            parent_frame,
            text="Capture Expression",
            command=self._capture_facial_expression,
            height=50,
            font=Fonts.BUTTON_MEDIUM,
            fg_color=ColorScheme.SECONDARY_GREEN
        )
        capture_btn.pack(pady=(0, 20))
    
    def _capture_facial_expression(self):
        """Capture facial expression"""
        # Simulate facial analysis
        # In real app, would capture image and analyze with AI
        self.befast_results["face"] = {"score": 0.85, "result": "Normal"}
        self._next_befast_step()
    
    def _show_arm_movement_interface(self, parent_frame):
        """Show arm movement interface"""
        # Instructions
        instructions_label = ctk.CTkLabel(
            parent_frame,
            text="Raise both arms in front of the camera and hold for 10 seconds.\nThe AI will detect any arm drift or weakness.",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_SECONDARY
        )
        instructions_label.pack(pady=(0, 20))
        
        # Camera preview for arm analysis
        self.arm_preview = ctk.CTkLabel(
            parent_frame,
            text="Camera Preview\nRaise your arms now",
            width=300,
            height=200,
            fg_color=ColorScheme.LIGHT_GRAY
        )
        self.arm_preview.pack(pady=20)
        
        # Start arm test button
        start_btn = ctk.CTkButton(
            parent_frame,
            text="Start Arm Test",
            command=self._start_arm_test,
            height=50,
            font=Fonts.BUTTON_MEDIUM,
            fg_color=ColorScheme.SECONDARY_GREEN
        )
        start_btn.pack(pady=(0, 20))
    
    def _start_arm_test(self):
        """Start arm movement test"""
        # Simulate arm test
        self.arm_preview.configure(text="Hold your arms steady...")
        
        # Simulate 10-second test
        self.after(3000, self._complete_arm_test)
    
    def _complete_arm_test(self):
        """Complete arm test"""
        self.befast_results["arms"] = {"score": 0.9, "result": "Normal"}
        self._next_befast_step()
    
    def _show_speech_interface(self, parent_frame):
        """Show speech interface"""
        # Instructions
        instructions_label = ctk.CTkLabel(
            parent_frame,
            text="Repeat the phrase clearly into your microphone.\nThe AI will analyze your speech for clarity and fluency.",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_SECONDARY
        )
        instructions_label.pack(pady=(0, 20))
        
        # Phrase to repeat
        phrase_label = ctk.CTkLabel(
            parent_frame,
            text="Repeat: 'The sky is blue and the sun is shining'",
            font=Fonts.BODY_LARGE,
            text_color=ColorScheme.PRIMARY_BLUE,
            wraplength=500
        )
        phrase_label.pack(pady=(0, 20))
        
        # Record button
        self.record_btn = ctk.CTkButton(
            parent_frame,
            text="Start Recording",
            command=self._start_speech_recording,
            height=60,
            font=Fonts.BUTTON_MEDIUM,
            fg_color=ColorScheme.SECONDARY_RED
        )
        self.record_btn.pack(pady=(0, 20))
    
    def _start_speech_recording(self):
        """Start speech recording"""
        self.record_btn.configure(text="Recording... Speak now", fg_color=ColorScheme.DANGER)
        
        # Simulate recording
        self.after(3000, self._complete_speech_recording)
    
    def _complete_speech_recording(self):
        """Complete speech recording"""
        self.record_btn.configure(text="Recording Complete", fg_color=ColorScheme.SUCCESS)
        
        # Simulate speech analysis
        self.befast_results["speech"] = {"score": 0.88, "result": "Normal"}
        
        self.after(1000, self._next_befast_step)
    
    def _show_generic_step_interface(self, parent_frame, step_type):
        """Show generic step interface"""
        # Instructions based on step type
        if step_type == "balance":
            instruction = "Stand with feet together and hold for 10 seconds.\nKeep your balance steady."
        elif step_type == "time":
            instruction = "Assessment timing completed.\nAll components have been evaluated."
        else:
            instruction = "Follow the instructions above."
        
        instruction_label = ctk.CTkLabel(
            parent_frame,
            text=instruction,
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_SECONDARY
        )
        instruction_label.pack(pady=(0, 30))
        
        # Complete button
        complete_btn = ctk.CTkButton(
            parent_frame,
            text="Complete Step",
            command=self._complete_generic_step,
            height=50,
            font=Fonts.BUTTON_MEDIUM,
            fg_color=ColorScheme.SECONDARY_GREEN
        )
        complete_btn.pack(pady=(0, 20))
    
    def _complete_generic_step(self):
        """Complete generic step"""
        step_name = ["balance", "eyes", "face", "arms", "speech", "time"][self.befast_step]
        self.befast_results[step_name] = {"score": 0.85, "result": "Normal"}
        self._next_befast_step()
    
    def _next_befast_step(self):
        """Move to next BE FAST step"""
        self.befast_step += 1
        self._show_befast_step()
    
    def _complete_befast_assessment(self):
        """Complete BE FAST assessment"""
        # Calculate overall score
        scores = [result["score"] for result in self.befast_results.values()]
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # Determine risk level
        if overall_score >= 0.8:
            risk_level = "Low"
        elif overall_score >= 0.6:
            risk_level = "Moderate"
        elif overall_score >= 0.4:
            risk_level = "High"
        else:
            risk_level = "Critical"
        
        # Create assessment result
        assessment_result = {
            "overall_score": overall_score,
            "risk_level": risk_level,
            "components": self.befast_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save assessment
        if self.backend.is_ready():
            self.backend.save_assessment(self.current_user_id, "befast", assessment_result)
        
        # Show results
        self._show_befast_results(assessment_result)
    
    def _show_befast_results(self, result):
        """Show BE FAST assessment results"""
        # Clear content
        for widget in self.assessment_area.winfo_children():
            widget.destroy()
        
        # Create results frame
        results_frame = ctk.CTkFrame(self.assessment_area)
        results_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            results_frame,
            text="BE FAST Assessment Results",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.pack(pady=(20, 20))
        
        # Overall score
        score_label = ctk.CTkLabel(
            results_frame,
            text=f"Overall Score: {result['overall_score']:.2f}",
            font=Fonts.BODY_LARGE,
            text_color=ColorScheme.TEXT_PRIMARY
        )
        score_label.pack(pady=(0, 10))
        
        # Risk level
        risk_color = self._get_risk_color_by_name(result['risk_level'])
        risk_label = ctk.CTkLabel(
            results_frame,
            text=f"Risk Level: {result['risk_level']}",
            font=Fonts.HEADING_SMALL,
            text_color=risk_color
        )
        risk_label.pack(pady=(0, 20))
        
        # Component results
        components_label = ctk.CTkLabel(
            results_frame,
            text="Component Results:",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_PRIMARY
        )
        components_label.pack(anchor="w", padx=50, pady=(0, 10))
        
        for component, data in result['components'].items():
            comp_label = ctk.CTkLabel(
                results_frame,
                text=f"• {component.title()}: {data['result']} ({data['score']:.2f})",
                font=Fonts.BODY_SMALL,
                text_color=ColorScheme.TEXT_SECONDARY
            )
            comp_label.pack(anchor="w", padx=70, pady=2)
        
        # Emergency response if high risk
        if result['risk_level'] in ['High', 'Critical']:
            self._trigger_emergency_response(result)
        
        # Continue button
        continue_btn = ctk.CTkButton(
            results_frame,
            text="Continue to Dashboard",
            command=self._return_to_dashboard,
            height=50,
            font=Fonts.BUTTON_MEDIUM,
            fg_color=ColorScheme.PRIMARY_BLUE
        )
        continue_btn.pack(pady=(30, 20))
        
        self.assessment_in_progress = False
    
    def _get_risk_color_by_name(self, risk_name: str) -> str:
        """Get color by risk name"""
        color_map = {
            "Low": ColorScheme.SUCCESS,
            "Moderate": ColorScheme.WARNING,
            "High": ColorScheme.DANGER,
            "Critical": ColorScheme.SECONDARY_RED
        }
        return color_map.get(risk_name, ColorScheme.TEXT_PRIMARY)
    
    def _trigger_emergency_response(self, assessment_result):
        """Trigger emergency response for high risk"""
        # Show emergency alert
        emergency_frame = ctk.CTkFrame(self.assessment_area, fg_color=ColorScheme.SECONDARY_RED)
        emergency_frame.pack(fill="x", padx=20, pady=10)
        
        emergency_label = ctk.CTkLabel(
            emergency_frame,
            text="⚠️ HIGH RISK DETECTED ⚠️\nEmergency services will be contacted automatically.",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.WHITE,
            justify="center"
        )
        emergency_label.pack(pady=20)
        
        # Trigger emergency alert
        symptoms = [comp for comp, data in assessment_result['components'].items() 
                   if data['result'] != 'Normal']
        
        alert = self.emergency_manager.trigger_emergency_alert(
            risk_level=assessment_result['risk_level'],
            symptoms=symptoms,
            assessment_data=assessment_result,
            user_id=self.current_user_id
        )
        
        # Show countdown
        self._show_emergency_countdown(15)
    
    def _show_emergency_countdown(self, seconds):
        """Show emergency countdown"""
        if seconds > 0:
            # Update countdown display
            countdown_label = ctk.CTkLabel(
                self.assessment_area,
                text=f"Emergency response in: {seconds} seconds",
                font=Fonts.BODY_MEDIUM,
                text_color=ColorScheme.SECONDARY_RED
            )
            countdown_label.pack(pady=10)
            
            # Schedule next countdown
            self.after(1000, lambda: self._show_emergency_countdown(seconds - 1))
        else:
            # Emergency response triggered
            messagebox.showinfo("Emergency Alert", "Emergency services have been contacted!")
    
    def _return_to_dashboard(self):
        """Return to main dashboard"""
        # Reset the assessment area so a new assessment can be started next time
        self._show_befast_assessment()
        self._show_view("dashboard")
    
    def _view_progress(self):
        """View progress (override from parent)"""
        self._show_view("progress")
    
    def _view_emergency(self):
        """View emergency contacts (override from parent)"""
        self._show_view("emergency")
    
    def _trigger_emergency(self):
        """Trigger emergency alert (override from parent)"""
        # Show emergency confirmation
        result = messagebox.askyesno(
            "Emergency Alert",
            "Are you experiencing stroke symptoms?\n\nThis will contact emergency services immediately.",
            icon="warning"
        )
        
        if result:
            # Trigger emergency alert
            alert = self.emergency_manager.trigger_emergency_alert(
                risk_level="Critical",
                symptoms=["Manual emergency trigger"],
                assessment_data={"manual_trigger": True},
                user_id=self.current_user_id
            )
            
            messagebox.showinfo("Emergency Alert", "Emergency services have been contacted!")
    
    def on_closing(self):
        """Handle application closing"""
        # Stop camera if active
        self._stop_camera()
        
        # Save user data
        if self.user_profile:
            with open(f"user_profile_{self.current_user_id}.json", "w") as f:
                json.dump(self.user_profile, f, indent=2)
        
        # Close application
        self.destroy()


def main():
    """Main application entry point"""
    # Create and run the application
    app = StrokeSenseMainApp()
    
    # Handle window closing
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Run the application
    app.mainloop()


if __name__ == "__main__":
    main()
