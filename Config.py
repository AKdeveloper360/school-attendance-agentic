"""
config.py - Central configuration for the AI Attendance Agent.

Every value can be overridden with an environment variable. On Streamlit
Community Cloud, top-level keys in "Secrets" are exposed as environment
variables, so you can set INSTITUTE_NAME, TIMEZONE, APP_PIN there.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Where reference photos live (format: Name_RollNumber.jpg)
FACES_DIR = BASE_DIR / "student_faces"

# Persistent attendance log (append-only; never overwritten)
ATTENDANCE_CSV = BASE_DIR / "attendance.csv"

# Report header
INSTITUTE_NAME = os.environ.get("INSTITUTE_NAME", "Your School / Institute Name")
INSTITUTE_SUBTITLE = os.environ.get("INSTITUTE_SUBTITLE", "Daily Attendance Report")

# Streamlit Cloud servers run in UTC, so we pin the timezone explicitly.
TIMEZONE = os.environ.get("TIMEZONE", "Asia/Karachi")

# Face matching strictness. Lower = stricter (fewer false matches).
# 0.6 is the library default; 0.5 is a safer value for attendance.
MATCH_TOLERANCE = float(os.environ.get("MATCH_TOLERANCE", "0.5"))

# Optional teacher PIN. Leave empty to disable the login gate.
APP_PIN = os.environ.get("APP_PIN", "")

# CSV schema
COLUMNS = ["Date", "Time", "Roll Number", "Student Name", "Status"]
DATE_FMT = "%Y-%m-%d"
TIME_FMT = "%H:%M:%S"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
