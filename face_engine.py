"""
face_engine.py - The Agent's "vision" module.

Responsibilities
  * Parse reference photos named  Name_RollNumber.jpg  into a Registry
  * Encode each reference face (128-d vector via dlib / face_recognition)
  * Detect + recognise every face in a new photo
  * Draw an annotated copy of the photo for the teacher
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageOps

from config import IMAGE_EXTS


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Student:
    roll: str
    name: str
    file: str


@dataclass
class Registry:
    """Everything the agent knows about enrolled students."""
    students: list[Student] = field(default_factory=list)          # all enrolled
    encoded_students: list[Student] = field(default_factory=list)  # usable for matching
    encodings: np.ndarray = field(default_factory=lambda: np.empty((0, 128)))
    problems: list[str] = field(default_factory=list)              # bad files / warnings


@dataclass
class Detection:
    box: tuple[int, int, int, int]       # (top, right, bottom, left) in original pixels
    student: Student | None = None
    distance: float | None = None
    status: str = "unknown"              # unknown | new | duplicate  (set by the app)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def roll_sort_key(roll: str):
    """Natural sort: 2 < 10 < 101 < A5."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", roll)]


def parse_filename(path: Path) -> Student | None:
    """'Ali_Khan_101.jpg' -> Student(roll='101', name='Ali Khan'). None if invalid."""
    stem = path.stem
    if "_" not in stem:
        return None
    name_part, roll = stem.rsplit("_", 1)
    name = name_part.replace("_", " ").strip()
    roll = roll.strip()
    if not name or not roll:
        return None
    return Student(roll=roll, name=name, file=path.name)


def load_image_bytes(raw: bytes, max_side: int | None = None) -> np.ndarray:
    """Decode bytes -> RGB uint8 array. Honors phone EXIF rotation."""
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_side:
        img.thumbnail((max_side, max_side))
    return np.ascontiguousarray(np.array(img))


def _load_image_path(path: Path, max_side: int = 1000) -> np.ndarray:
    return load_image_bytes(path.read_bytes(), max_side=max_side)


def encode_reference(rgb: np.ndarray):
    """Return (encoding | None, number_of_faces_found). Uses the largest face."""
    locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
    if not locations:
        return None, 0
    largest = max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
    encoding = face_recognition.face_encodings(rgb, [largest], num_jitters=1)[0]
    return encoding, len(locations)


def faces_signature(faces_dir: Path) -> tuple:
    """Cheap fingerprint of the folder - lets Streamlit know when to re-encode."""
    if not faces_dir.exists():
        return ()
    return tuple(sorted(
        (p.name, p.stat().st_mtime_ns, p.stat().st_size)
        for p in faces_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    ))


# --------------------------------------------------------------------------- #
# Registry loading
# --------------------------------------------------------------------------- #
def load_registry(faces_dir: Path) -> Registry:
    """Scan student_faces/, parse names, and encode every reference face."""
    faces_dir.mkdir(parents=True, exist_ok=True)
    reg = Registry()
    seen_rolls: set[str] = set()
    encodings: list[np.ndarray] = []

    for path in sorted(faces_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        student = parse_filename(path)
        if student is None:
            reg.problems.append(f"{path.name}: invalid name - expected Name_RollNumber.jpg")
            continue
        if student.roll in seen_rolls:
            reg.problems.append(f"{path.name}: duplicate roll number {student.roll} - ignored")
            continue
        seen_rolls.add(student.roll)
        reg.students.append(student)          # enrolled even if the face can't be encoded

        try:
            rgb = _load_image_path(path)
            encoding, n_faces = encode_reference(rgb)
        except Exception as exc:              # corrupt image etc.
            reg.problems.append(f"{path.name}: could not read image ({exc})")
            continue

        if encoding is None:
            reg.problems.append(f"{path.name}: no face detected - use a clearer, front-facing photo")
            continue
        if n_faces > 1:
            reg.problems.append(f"{path.name}: {n_faces} faces found - used the largest one")

        encodings.append(encoding)
        reg.encoded_students.append(student)

    reg.students.sort(key=lambda s: roll_sort_key(s.roll))
    if encodings:
        reg.encodings = np.array(encodings)
    return reg


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #
def recognize_faces(rgb: np.ndarray, registry: Registry,
                    tolerance: float = 0.5, max_width: int = 1600,
                    upsample: int = 1) -> list[Detection]:
    """
    Detect every face in `rgb` and match each to the closest enrolled student.

    * Large photos are downscaled for speed; boxes are mapped back afterwards.
    * If two faces claim the same student, only the closest match keeps it
      (prevents one student being "recognised" twice in a single photo).
    """
    h, w = rgb.shape[:2]
    scale = min(1.0, max_width / float(w))
    small = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
    small = np.ascontiguousarray(small)

    locations = face_recognition.face_locations(
        small, number_of_times_to_upsample=upsample, model="hog")
    encodings = face_recognition.face_encodings(small, locations)

    detections = [
        Detection(box=tuple(int(v / scale) for v in loc)) for loc in locations
    ]
    if not detections or len(registry.encodings) == 0:
        return detections

    candidates = []  # (detection_idx, student_idx, distance)
    for i, enc in enumerate(encodings):
        distances = face_recognition.face_distance(registry.encodings, enc)
        j = int(np.argmin(distances))
        if distances[j] <= tolerance:
            candidates.append((i, j, float(distances[j])))

    claimed: set[int] = set()
    for i, j, dist in sorted(candidates, key=lambda c: c[2]):   # best match first
        if j in claimed:
            continue
        claimed.add(j)
        detections[i].student = registry.encoded_students[j]
        detections[i].distance = dist
    return detections


# --------------------------------------------------------------------------- #
# Annotation
# --------------------------------------------------------------------------- #
_COLORS = {                      # RGB
    "new": (0, 190, 80),         # green  - newly marked present
    "duplicate": (255, 150, 0),  # orange - already marked today
    "unknown": (220, 40, 40),    # red    - not recognised
}


def annotate_image(rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of the photo with labelled boxes."""
    out = rgb.copy()
    h, w = out.shape[:2]
    thickness = max(2, w // 400)
    font_scale = max(0.5, w / 1400)

    for det in detections:
        top, right, bottom, left = det.box
        color = _COLORS.get(det.status, _COLORS["unknown"])
        cv2.rectangle(out, (left, top), (right, bottom), color, thickness)

        label = f"{det.student.name} ({det.student.roll})" if det.student else "Unknown"
        label = label.encode("ascii", "replace").decode()  # OpenCV fonts are ASCII-only
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        y0 = max(0, top - th - base - 6)
        cv2.rectangle(out, (left, y0), (left + tw + 8, y0 + th + base + 6), color, -1)
        cv2.putText(out, label, (left + 4, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out
