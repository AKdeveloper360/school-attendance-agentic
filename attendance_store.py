"""
attendance_store.py - Persistent attendance log + agent decision filters.

* Append-only CSV: history is never rewritten or overwritten.
* Duplicate filter: one "Present" row per (Date, Roll Number).
* Absentee filter: enrolled students minus today's present set.
"""
from __future__ import annotations

import csv
import io
import threading
from datetime import datetime

import pandas as pd

from config import ATTENDANCE_CSV, COLUMNS, DATE_FMT, TIME_FMT
from face_engine import Student, roll_sort_key

# One lock per process: protects the check-then-append sequence.
_lock = threading.Lock()


def _ensure_file() -> None:
    if not ATTENDANCE_CSV.exists():
        # utf-8-sig so Excel opens non-English names correctly
        with open(ATTENDANCE_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(COLUMNS)


def _already_marked(date_str: str, roll: str) -> bool:
    with open(ATTENDANCE_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("Date") == date_str and (row.get("Roll Number") or "").strip() == roll:
                return True
    return False


def mark_present(student: Student, now: datetime) -> bool:
    """
    Log attendance. Returns True if a new row was written,
    False if the student was already marked present on that date (duplicate rejected).
    """
    date_str = now.strftime(DATE_FMT)
    with _lock:
        _ensure_file()
        if _already_marked(date_str, student.roll):
            return False
        with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [date_str, now.strftime(TIME_FMT), student.roll, student.name, "Present"])
    return True


def load_attendance(date_str: str | None = None) -> pd.DataFrame:
    """Read the log (optionally one date). Always returns the standard columns."""
    with _lock:
        _ensure_file()
        df = pd.read_csv(ATTENDANCE_CSV, dtype=str, keep_default_na=False,
                         encoding="utf-8-sig")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[COLUMNS]
    if date_str:
        df = df[df["Date"] == date_str]
    return df.sort_values(["Date", "Time"]).reset_index(drop=True)


def compute_absentees(students: list[Student], present_df: pd.DataFrame) -> list[Student]:
    """Absentees Filter: every enrolled student not in the present list."""
    present_rolls = set(present_df["Roll Number"].str.strip())
    absent = [s for s in students if s.roll not in present_rolls]
    return sorted(absent, key=lambda s: roll_sort_key(s.roll))


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Attendance") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()
