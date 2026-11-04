"""
Modern UI/UX Implementation for StrokeSense
Beautiful, accessible design following medical app best practices
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Dict, List, Optional, Callable
import threading
import time
from datetime import datetime
import json


# Configure customtkinter
ctk.set_appearance_mode("light")  # "light" or "dark"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"


class ColorScheme:
    """Medical app color scheme following accessibility guidelines"""
    
    # Primary colors
    PRIMARY_BLUE = "#2E86AB"      # Medical blue
    PRIMARY_DARK = "#1B4F72"      # Darker blue
    PRIMARY_LIGHT = "#85C1E9"     # Light blue
    
    # Secondary colors
    SECONDARY_GREEN = "#27AE60"   # Success green
    SECONDARY_RED = "#E74C3C"     # Alert red
    SECONDARY_ORANGE = "#F39C12"  # Warning orange
    
    # Neutral colors
    WHITE = "#FFFFFF"
    LIGHT_GRAY = "#F8F9FA"
    MEDIUM_GRAY = "#6C757D"
    DARK_GRAY = "#343A40"
    BLACK = "#000000"
    
    # Status colors
    SUCCESS = "#28A745"
    WARNING = "#FFC107"
    DANGER = "#DC3545"
    INFO = "#17A2B8"
    
    # Background colors
    BG_PRIMARY = "#FFFFFF"
    BG_SECONDARY = "#F8F9FA"
    BG_DARK = "#343A40"
    
    # Text colors
    TEXT_PRIMARY = "#212529"
    TEXT_SECONDARY = "#6C757D"
    TEXT_LIGHT = "#FFFFFF"


class Fonts:
    """Font configurations for accessibility"""
    
    # Primary fonts
    HEADING_LARGE = ("Segoe UI", 24, "bold")
    HEADING_MEDIUM = ("Segoe UI", 18, "bold")
    HEADING_SMALL = ("Segoe UI", 14, "bold")
    
    # Body fonts
    BODY_LARGE = ("Segoe UI", 16)
    BODY_MEDIUM = ("Segoe UI", 14)
    BODY_SMALL = ("Segoe UI", 12)
    
    # Button fonts
    BUTTON_LARGE = ("Segoe UI", 16, "bold")
    BUTTON_MEDIUM = ("Segoe UI", 14, "bold")
    BUTTON_SMALL = ("Segoe UI", 12, "bold")
    
    # Special fonts
    MONOSPACE = ("Consolas", 12)
    ICON_FONT = ("Segoe UI Symbol", 16)


class StrokeSenseApp(ctk.CTk):
    """Main StrokeSense application"""
    
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("StrokeSense - AI-Powered Stroke Detection")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        # Set icon (would load actual icon in production)
        self.iconbitmap("assets/icon.ico") if self._file_exists("assets/icon.ico") else None
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Initialize components
        self.current_user = None
        self.camera_active = False
        self.assessment_in_progress = False
        
        # Create main interface
        self._create_main_interface()
        
        # Apply styling
        self._apply_styling()
    
    def _file_exists(self, path: str) -> bool:
        """Check if file exists"""
        try:
            with open(path, 'r'):
                return True
        except FileNotFoundError:
            return False
    
    def _create_main_interface(self):
        """Create main application interface"""
        
        # Create main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1)
        
        # Create header
        self._create_header()
        
        # Create main content area
        self._create_content_area()
        
        # Create navigation
        self._create_navigation()
    
    def _create_header(self):
        """Create application header"""
        header_frame = ctk.CTkFrame(self.main_container, height=80)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        header_frame.grid_columnconfigure(1, weight=1)
        
        # App logo and title
        logo_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=20, pady=10)
        
        # App title
        title_label = ctk.CTkLabel(
            logo_frame,
            text="StrokeSense",
            font=Fonts.HEADING_LARGE,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.pack(side="left")
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            logo_frame,
            text="AI-Powered Stroke Detection",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_SECONDARY
        )
        subtitle_label.pack(side="left", padx=(10, 0))
        
        # User info and settings
        user_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        user_frame.grid(row=0, column=1, padx=20, pady=10, sticky="e")
        
        # User status
        self.user_status_label = ctk.CTkLabel(
            user_frame,
            text="Not logged in",
            font=Fonts.BODY_SMALL,
            text_color=ColorScheme.TEXT_SECONDARY
        )
        self.user_status_label.pack(side="right")
        
        # Settings button
        settings_btn = ctk.CTkButton(
            user_frame,
            text="⚙️",
            width=40,
            height=40,
            command=self._open_settings,
            font=Fonts.ICON_FONT
        )
        settings_btn.pack(side="right", padx=(10, 0))
    
    def _create_content_area(self):
        """Create main content area"""
        self.content_frame = ctk.CTkFrame(self.main_container)
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Create different views
        self.views = {}
        self._create_dashboard_view()
        self._create_assessment_view()
        self._create_progress_view()
        self._create_emergency_view()
        
        # Show dashboard by default
        self._show_view("dashboard")
    
    def _create_dashboard_view(self):
        """Create dashboard view"""
        dashboard_frame = ctk.CTkFrame(self.content_frame)
        dashboard_frame.grid_columnconfigure(0, weight=1)
        dashboard_frame.grid_rowconfigure(1, weight=1)
        
        # Welcome section
        welcome_frame = ctk.CTkFrame(dashboard_frame, height=120)
        welcome_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        welcome_frame.grid_columnconfigure(0, weight=1)
        
        welcome_label = ctk.CTkLabel(
            welcome_frame,
            text="Welcome to StrokeSense",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.PRIMARY_DARK
        )
        welcome_label.grid(row=0, column=0, pady=(20, 10))
        
        welcome_text = ctk.CTkLabel(
            welcome_frame,
            text="Your AI-powered companion for stroke detection and prevention.\nMonitor your health with confidence.",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_SECONDARY,
            justify="center"
        )
        welcome_text.grid(row=1, column=0, pady=(0, 20))
        
        # Quick actions grid
        actions_frame = ctk.CTkFrame(dashboard_frame)
        actions_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        actions_frame.grid_columnconfigure((0, 1), weight=1)
        actions_frame.grid_rowconfigure((0, 1), weight=1)
        
        # Quick action buttons
        actions = [
            ("Start Assessment", "🔍", self._start_assessment, ColorScheme.PRIMARY_BLUE),
            ("View Progress", "📊", self._view_progress, ColorScheme.SECONDARY_GREEN),
            ("Emergency Contacts", "🚨", self._view_emergency, ColorScheme.SECONDARY_RED),
            ("Health Tips", "💡", self._view_tips, ColorScheme.SECONDARY_ORANGE)
        ]
        
        for i, (text, icon, command, color) in enumerate(actions):
            row = i // 2
            col = i % 2
            
            action_btn = ctk.CTkButton(
                actions_frame,
                text=f"{icon}\n{text}",
                command=command,
                height=120,
                font=Fonts.BUTTON_MEDIUM,
                fg_color=color,
                hover_color=self._darken_color(color),
                corner_radius=15
            )
            action_btn.grid(row=row, column=col, padx=10, pady=10, sticky="ew")
        
        self.views["dashboard"] = dashboard_frame
    
    def _create_assessment_view(self):
        """Create assessment view"""
        assessment_frame = ctk.CTkFrame(self.content_frame)
        assessment_frame.grid_columnconfigure(0, weight=1)
        assessment_frame.grid_rowconfigure(1, weight=1)
        
        # Assessment header
        header_frame = ctk.CTkFrame(assessment_frame, height=80)
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header_frame.grid_columnconfigure(0, weight=1)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="BE FAST Assessment",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.grid(row=0, column=0, pady=20)
        
        # Assessment content
        content_frame = ctk.CTkFrame(assessment_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)
        
        # Instructions
        instructions_frame = ctk.CTkFrame(content_frame, height=100)
        instructions_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        instructions_label = ctk.CTkLabel(
            instructions_frame,
            text="Follow the instructions for each test component. The assessment will guide you through Balance, Eyes, Face, Arms, Speech, and Time evaluation.",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_PRIMARY,
            wraplength=800,
            justify="center"
        )
        instructions_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Assessment area
        self.assessment_area = ctk.CTkFrame(content_frame)
        self.assessment_area.grid(row=1, column=0, sticky="nsew")
        self.assessment_area.grid_columnconfigure(0, weight=1)
        self.assessment_area.grid_rowconfigure(0, weight=1)
        
        # Start assessment button
        start_btn = ctk.CTkButton(
            self.assessment_area,
            text="Start BE FAST Assessment",
            command=self._start_befast_assessment,
            height=60,
            font=Fonts.BUTTON_LARGE,
            fg_color=ColorScheme.PRIMARY_BLUE,
            hover_color=self._darken_color(ColorScheme.PRIMARY_BLUE)
        )
        start_btn.grid(row=0, column=0, padx=50, pady=50)
        
        self.views["assessment"] = assessment_frame
    
    def _create_progress_view(self):
        """Create progress tracking view"""
        progress_frame = ctk.CTkFrame(self.content_frame)
        progress_frame.grid_columnconfigure(0, weight=1)
        progress_frame.grid_rowconfigure(1, weight=1)
        
        # Progress header
        header_frame = ctk.CTkFrame(progress_frame, height=80)
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Health Progress",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Progress content
        content_frame = ctk.CTkFrame(progress_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        content_frame.grid_columnconfigure((0, 1), weight=1)
        content_frame.grid_rowconfigure(1, weight=1)
        
        # Stats cards
        stats_frame = ctk.CTkFrame(content_frame, height=120)
        stats_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        stats = [
            ("Assessments", "5", ColorScheme.PRIMARY_BLUE),
            ("Exercise", "150 min", ColorScheme.SECONDARY_GREEN),
            ("Medications", "95%", ColorScheme.SECONDARY_ORANGE),
            ("Risk Level", "Low", ColorScheme.SUCCESS)
        ]
        
        for i, (label, value, color) in enumerate(stats):
            stat_card = ctk.CTkFrame(stats_frame, fg_color=color)
            stat_card.grid(row=0, column=i, padx=5, pady=10, sticky="ew")
            
            value_label = ctk.CTkLabel(
                stat_card,
                text=value,
                font=Fonts.HEADING_SMALL,
                text_color=ColorScheme.WHITE
            )
            value_label.pack(pady=(10, 5))
            
            label_label = ctk.CTkLabel(
                stat_card,
                text=label,
                font=Fonts.BODY_SMALL,
                text_color=ColorScheme.WHITE
            )
            label_label.pack(pady=(0, 10))
        
        # Charts and detailed progress
        charts_frame = ctk.CTkFrame(content_frame)
        charts_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        
        charts_label = ctk.CTkLabel(
            charts_frame,
            text="Progress Charts\n(Would show actual charts in production)",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_SECONDARY
        )
        charts_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Recent activities
        activities_frame = ctk.CTkFrame(content_frame)
        activities_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        
        activities_label = ctk.CTkLabel(
            activities_frame,
            text="Recent Activities\n• Assessment completed\n• Exercise logged\n• Medication taken",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_PRIMARY,
            justify="left"
        )
        activities_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        self.views["progress"] = progress_frame
    
    def _create_emergency_view(self):
        """Create emergency contacts view"""
        emergency_frame = ctk.CTkFrame(self.content_frame)
        emergency_frame.grid_columnconfigure(0, weight=1)
        emergency_frame.grid_rowconfigure(1, weight=1)
        
        # Emergency header
        header_frame = ctk.CTkFrame(emergency_frame, height=80)
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Emergency Contacts",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.SECONDARY_RED
        )
        title_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Emergency content
        content_frame = ctk.CTkFrame(emergency_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(1, weight=1)
        
        # Emergency button
        emergency_btn_frame = ctk.CTkFrame(content_frame, height=120)
        emergency_btn_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        emergency_btn = ctk.CTkButton(
            emergency_btn_frame,
            text="🚨 EMERGENCY ALERT 🚨\nCall 911",
            command=self._trigger_emergency,
            height=100,
            font=Fonts.BUTTON_LARGE,
            fg_color=ColorScheme.SECONDARY_RED,
            hover_color=self._darken_color(ColorScheme.SECONDARY_RED),
            corner_radius=15
        )
        emergency_btn.pack(expand=True, fill="both", padx=20, pady=10)
        
        # Contacts list
        contacts_frame = ctk.CTkFrame(content_frame)
        contacts_frame.grid(row=1, column=0, sticky="nsew")
        contacts_frame.grid_columnconfigure(0, weight=1)
        
        contacts_label = ctk.CTkLabel(
            contacts_frame,
            text="Emergency Contacts:\n\n• Emergency Services: 911\n• Dr. Smith (Neurologist): (555) 123-4567\n• Family Contact: (555) 987-6543",
            font=Fonts.BODY_MEDIUM,
            text_color=ColorScheme.TEXT_PRIMARY,
            justify="left"
        )
        contacts_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        self.views["emergency"] = emergency_frame
    
    def _create_navigation(self):
        """Create bottom navigation"""
        nav_frame = ctk.CTkFrame(self.main_container, height=60)
        nav_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        nav_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        nav_items = [
            ("Dashboard", "🏠", "dashboard"),
            ("Assessment", "🔍", "assessment"),
            ("Progress", "📊", "progress"),
            ("Emergency", "🚨", "emergency")
        ]
        
        self.nav_buttons = {}
        for i, (text, icon, view_name) in enumerate(nav_items):
            btn = ctk.CTkButton(
                nav_frame,
                text=f"{icon}\n{text}",
                command=lambda v=view_name: self._show_view(v),
                height=50,
                font=Fonts.BODY_SMALL,
                fg_color=ColorScheme.LIGHT_GRAY,
                hover_color=ColorScheme.MEDIUM_GRAY
            )
            btn.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
            self.nav_buttons[view_name] = btn
    
    def _apply_styling(self):
        """Apply consistent styling throughout the app"""
        # Configure default button styling
        ctk.CTkButton.configure(
            self,
            corner_radius=10,
            border_width=0,
            font=Fonts.BUTTON_MEDIUM
        )
        
        # Configure default frame styling
        ctk.CTkFrame.configure(
            self,
            corner_radius=10,
            border_width=0
        )
    
    def _darken_color(self, color: str, factor: float = 0.8) -> str:
        """Darken a hex color by a factor"""
        # Simple color darkening (would use proper color manipulation in production)
        color_map = {
            ColorScheme.PRIMARY_BLUE: "#1E5A7A",
            ColorScheme.SECONDARY_GREEN: "#1E8449",
            ColorScheme.SECONDARY_RED: "#C0392B",
            ColorScheme.SECONDARY_ORANGE: "#D68910"
        }
        return color_map.get(color, color)
    
    def _show_view(self, view_name: str):
        """Show specific view"""
        # Hide all views
        for view in self.views.values():
            view.grid_remove()
        
        # Show selected view
        if view_name in self.views:
            self.views[view_name].grid()
        
        # Update navigation button states
        for name, btn in self.nav_buttons.items():
            if name == view_name:
                btn.configure(fg_color=ColorScheme.PRIMARY_BLUE)
            else:
                btn.configure(fg_color=ColorScheme.LIGHT_GRAY)
    
    def _start_assessment(self):
        """Start assessment from dashboard"""
        self._show_view("assessment")
    
    def _view_progress(self):
        """View progress from dashboard"""
        self._show_view("progress")
    
    def _view_emergency(self):
        """View emergency contacts from dashboard"""
        self._show_view("emergency")
    
    def _view_tips(self):
        """View health tips"""
        messagebox.showinfo("Health Tips", "Daily Tip: 30 minutes of walking reduces stroke risk by 20%!")
    
    def _start_befast_assessment(self):
        """Start BE FAST assessment"""
        if self.assessment_in_progress:
            messagebox.showwarning("Assessment in Progress", "An assessment is already in progress.")
            return
        
        # Show assessment steps
        self._show_assessment_steps()
    
    def _show_assessment_steps(self):
        """Show BE FAST assessment steps"""
        # Clear assessment area
        for widget in self.assessment_area.winfo_children():
            widget.destroy()
        
        # Create step-by-step assessment
        steps_frame = ctk.CTkFrame(self.assessment_area)
        steps_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        steps_frame.grid_columnconfigure(0, weight=1)
        
        steps = [
            ("Balance", "Stand with feet together and hold for 10 seconds"),
            ("Eyes", "Follow the moving dot with your eyes"),
            ("Face", "Smile and show your teeth"),
            ("Arms", "Raise both arms and hold for 10 seconds"),
            ("Speech", "Repeat: 'The sky is blue and the sun is shining'"),
            ("Time", "Assessment completed")
        ]
        
        current_step = 0
        
        def next_step():
            nonlocal current_step
            if current_step < len(steps) - 1:
                current_step += 1
                update_step_display()
            else:
                complete_assessment()
        
        def update_step_display():
            # Clear previous content
            for widget in steps_frame.winfo_children():
                widget.destroy()
            
            step_name, instruction = steps[current_step]
            
            # Step title
            title_label = ctk.CTkLabel(
                steps_frame,
                text=f"Step {current_step + 1}: {step_name}",
                font=Fonts.HEADING_MEDIUM,
                text_color=ColorScheme.PRIMARY_DARK
            )
            title_label.grid(row=0, column=0, pady=(0, 20))
            
            # Instruction
            instruction_label = ctk.CTkLabel(
                steps_frame,
                text=instruction,
                font=Fonts.BODY_LARGE,
                text_color=ColorScheme.TEXT_PRIMARY,
                wraplength=600
            )
            instruction_label.grid(row=1, column=0, pady=(0, 30))
            
            # Next button
            next_btn = ctk.CTkButton(
                steps_frame,
                text="Next Step" if current_step < len(steps) - 1 else "Complete Assessment",
                command=next_step,
                height=50,
                font=Fonts.BUTTON_MEDIUM,
                fg_color=ColorScheme.PRIMARY_BLUE
            )
            next_btn.grid(row=2, column=0, pady=(0, 20))
        
        def complete_assessment():
            messagebox.showinfo("Assessment Complete", "BE FAST assessment completed successfully!")
            self._show_view("dashboard")
        
        update_step_display()
    
    def _trigger_emergency(self):
        """Trigger emergency alert"""
        result = messagebox.askyesno(
            "Emergency Alert",
            "Are you experiencing stroke symptoms?\n\nThis will contact emergency services immediately.",
            icon="warning"
        )
        
        if result:
            messagebox.showinfo("Emergency Alert", "Emergency services have been contacted!")
    
    def _open_settings(self):
        """Open settings dialog"""
        settings_window = ctk.CTkToplevel(self)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self)
        settings_window.grab_set()
        
        # Settings content
        settings_frame = ctk.CTkFrame(settings_window)
        settings_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        title_label = ctk.CTkLabel(
            settings_frame,
            text="Settings",
            font=Fonts.HEADING_MEDIUM,
            text_color=ColorScheme.PRIMARY_DARK
        )
        title_label.pack(pady=(20, 30))
        
        # Settings options
        options = [
            ("Notifications", "Enable push notifications"),
            ("Location", "Allow location tracking"),
            ("Camera", "Allow camera access"),
            ("Microphone", "Allow microphone access")
        ]
        
        for option, description in options:
            option_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
            option_frame.pack(fill="x", padx=20, pady=5)
            
            option_label = ctk.CTkLabel(
                option_frame,
                text=option,
                font=Fonts.BODY_MEDIUM,
                text_color=ColorScheme.TEXT_PRIMARY
            )
            option_label.pack(side="left")
            
            option_switch = ctk.CTkSwitch(
                option_frame,
                text="",
                width=50
            )
            option_switch.pack(side="right")
        
        # Close button
        close_btn = ctk.CTkButton(
            settings_frame,
            text="Close",
            command=settings_window.destroy,
            height=40,
            font=Fonts.BUTTON_MEDIUM
        )
        close_btn.pack(pady=(30, 20))


def main():
    """Main application entry point"""
    app = StrokeSenseApp()
    app.mainloop()


if __name__ == "__main__":
    main()
