
"""
app.py - AI Attendance Agent (Streamlit front-end + agent loop)

Agent loop for every photo:
  PERCEIVE  -> decode image, detect faces
  DECIDE    -> match against known encodings, apply duplicate filter
  ACT       -> append attendance row, compute absentees, build reports
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from PIL import Image, ImageOps

import config
from attendance_store import (compute_absentees, load_attendance, mark_present,
                              to_excel_bytes)
from face_engine import (annotate_image, encode_reference, faces_signature,
                         load_image_bytes, load_registry, recognize_faces)
from report import build_pdf_report

# --------------------------------------------------------------------------- #
# Page setup + mobile-friendly styling
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AI Attendance Agent", page_icon="🎓",
                   layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .block-container {padding: 1rem 0.8rem 3rem 0.8rem; max-width: 760px;}
  .stButton > button, .stDownloadButton > button {
      width: 100%; min-height: 3rem; font-size: 1.05rem; border-radius: 12px;}
  [data-testid="stMetricValue"] {font-size: 1.8rem;}
  [data-testid="stCameraInput"] button {min-height: 3rem;}
  h1 {font-size: 1.6rem !important;}
</style>
""", unsafe_allow_html=True)


def now_local() -> datetime:
    return datetime.now(ZoneInfo(config.TIMEZONE))


# --------------------------------------------------------------------------- #
# Optional PIN gate (set APP_PIN in Streamlit secrets / env to enable)
# --------------------------------------------------------------------------- #
if config.APP_PIN and not st.session_state.get("auth"):
    st.title("🎓 AI Attendance Agent")
    pin = st.text_input("Teacher PIN", type="password")
    if pin and pin == config.APP_PIN:
        st.session_state["auth"] = True
        st.rerun()
    elif pin:
        st.error("Incorrect PIN.")
    st.stop()


# --------------------------------------------------------------------------- #
# Agent memory: encoded student faces (re-built only when the folder changes)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="🧠 Agent is learning student faces…")
def get_registry(_signature):
    return load_registry(config.FACES_DIR)


registry = get_registry(faces_signature(config.FACES_DIR))

# --------------------------------------------------------------------------- #
# Sidebar settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    institute = st.text_input("Institute name", config.INSTITUTE_NAME)
    tolerance = st.slider("Match strictness (lower = stricter)", 0.35, 0.65,
                          config.MATCH_TOLERANCE, 0.01,
                          help="0.50 is a good start. Raise slightly if real students "
                               "are shown as Unknown; lower it if wrong matches occur.")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🎓 AI Attendance Agent")
now = now_local()
today_str = now.strftime(config.DATE_FMT)

if st.session_state.get("flash"):
    st.success(st.session_state.pop("flash"))

if not registry.students:
    st.warning("No students enrolled yet. Add photos named `Name_RollNumber.jpg` to the "
               "`student_faces/` folder, or use the **Students** tab to enroll someone.")


# --------------------------------------------------------------------------- #
# Agent core: process one photo
# --------------------------------------------------------------------------- #
def process_photo(raw: bytes, label: str) -> dict:
    """Perceive -> Decide -> Act for a single image."""
    rgb = load_image_bytes(raw)
    detections = recognize_faces(rgb, registry, tolerance=tolerance)
    stamp = now_local()

    new, duplicates, unknown = [], [], 0
    for det in detections:
        if det.student is None:
            det.status = "unknown"
            unknown += 1
        elif mark_present(det.student, stamp):          # duplicate filter lives here
            det.status = "new"
            new.append((det.student, stamp.strftime("%H:%M:%S")))
        else:
            det.status = "duplicate"
            duplicates.append(det.student)

    return {"label": label, "faces": len(detections), "new": new,
            "duplicates": duplicates, "unknown": unknown,
            "image": annotate_image(rgb, detections)}


def render_result(res: dict) -> None:
    st.image(res["image"], caption=res["label"], use_container_width=True)
    if res["faces"] == 0:
        st.warning("No faces detected. Move closer, face the light, and try again.")
        return
    for s, t in res["new"]:
        st.success(f"✅ **{s.name}** (Roll {s.roll}) marked **Present** at {t}")
    for s in res["duplicates"]:
        st.info(f"ℹ️ **{s.name}** (Roll {s.roll}) already marked today - duplicate rejected")
    if res["unknown"]:
        st.warning(f"❓ {res['unknown']} face(s) not recognised (not enrolled, "
                   "too small/blurry, or match too weak).")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_scan, tab_today, tab_reports, tab_students = st.tabs(
    ["📷 Scan", "📋 Attendance", "📄 Reports", "👥 Students"])

# ============================== SCAN ======================================== #
with tab_scan:
    present_today = load_attendance(today_str)
    st.caption(f"📅 {now:%A, %d %B %Y} · {len(present_today)} of {len(registry.students)} present")

    mode = st.radio("Input", ["📷 Camera", "🖼️ Upload photo(s)"], horizontal=True,
                    label_visibility="collapsed")

    if mode == "📷 Camera":
        shot = st.camera_input("Capture Class Photo / Scan Face")
        if shot is not None:
            raw = shot.getvalue()
            digest = hashlib.md5(raw).hexdigest()
            if st.session_state.get("last_cam") != digest:      # avoid re-processing on rerun
                with st.spinner("🔍 Agent is analysing faces…"):
                    st.session_state["results"] = [process_photo(raw, "Camera capture")]
                st.session_state["last_cam"] = digest
        else:
            st.session_state.pop("last_cam", None)
    else:
        files = st.file_uploader("Class photo(s)", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True)
        if files and st.button("🚀 Run Agent on photo(s)", type="primary"):
            results = []
            with st.spinner("🔍 Agent is analysing faces…"):
                for f in files:
                    results.append(process_photo(f.getvalue(), f.name))
            st.session_state["results"] = results

    for res in st.session_state.get("results", []):
        st.divider()
        render_result(res)

# ============================== ATTENDANCE ================================== #
with tab_today:
    view_date = st.date_input("Date", value=now.date(), max_value=now.date())
    vd = view_date.strftime(config.DATE_FMT)
    day_df = load_attendance(vd)
    absent = compute_absentees(registry.students, day_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Enrolled", len(registry.students))
    c2.metric("Present", len(day_df))
    c3.metric("Absent", len(absent))

    view = st.radio("Show", ["✅ Present", "❌ Absent"], horizontal=True)
    if view.startswith("✅"):
        if day_df.empty:
            st.info("Nobody marked present for this date yet.")
        else:
            st.dataframe(day_df, use_container_width=True, hide_index=True)
    else:
        if not absent:
            st.success("No absentees 🎉" if registry.students else "No students enrolled.")
        else:
            st.dataframe(
                [{"Roll Number": s.roll, "Student Name": s.name, "Status": "Absent"}
                 for s in absent], use_container_width=True, hide_index=True)

# ============================== REPORTS ===================================== #
with tab_reports:
    rep_date = st.date_input("Report date", value=now.date(), max_value=now.date(),
                             key="rep_date")
    rd = rep_date.strftime(config.DATE_FMT)
    rep_df = load_attendance(rd)
    rep_absent = compute_absentees(registry.students, rep_df)

    pdf_bytes = build_pdf_report(institute, config.INSTITUTE_SUBTITLE, rep_date,
                                 rep_df, rep_absent, len(registry.students), now_local())
    st.download_button("📄 Export PDF Report", pdf_bytes,
                       file_name=f"attendance_{rd}.pdf", mime="application/pdf",
                       type="primary")

    st.divider()
    st.subheader("Full attendance history")
    history = load_attendance()
    st.caption(f"{len(history)} records · history is never overwritten")
    st.dataframe(history, use_container_width=True, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ CSV", history.to_csv(index=False).encode("utf-8-sig"),
                       "attendance.csv", "text/csv")
    d2.download_button("⬇️ Excel", to_excel_bytes(history), "attendance.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption("💡 On Streamlit Community Cloud the server disk resets on restart - "
               "download a backup regularly.")

# ============================== STUDENTS ==================================== #
with tab_students:
    st.subheader(f"Enrolled students ({len(registry.students)})")
    encoded_rolls = {s.roll for s in registry.encoded_students}
    st.dataframe(
        [{"Roll Number": s.roll, "Name": s.name,
          "Face ready": "✅" if s.roll in encoded_rolls else "⚠️ no face found"}
         for s in registry.students],
        use_container_width=True, hide_index=True)

    if registry.problems:
        with st.expander(f"⚠️ {len(registry.problems)} photo issue(s)"):
            for p in registry.problems:
                st.write("• " + p)

    st.divider()
    st.subheader("➕ Enroll a new student")
    with st.form("enroll", clear_on_submit=True):
        name = st.text_input("Full name")
        roll = st.text_input("Roll number")
        photo = st.file_uploader("Clear, front-facing photo", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Enroll student")

    if submitted:
        clean_name = re.sub(r"[^\w ]", "", name, flags=re.UNICODE).strip().replace(" ", "_")
        clean_roll = re.sub(r"[^A-Za-z0-9-]", "", roll)
        if not (clean_name and clean_roll and photo):
            st.error("Name, roll number and photo are all required.")
        elif clean_roll in {s.roll for s in registry.students}:
            st.error(f"Roll number {clean_roll} already exists.")
        else:
            rgb = load_image_bytes(photo.getvalue(), max_side=1000)
            _, n_faces = encode_reference(rgb)
            if n_faces != 1:
                st.error(f"Found {n_faces} faces - the photo must contain exactly one face.")
            else:
                config.FACES_DIR.mkdir(exist_ok=True)
                Image.fromarray(rgb).save(config.FACES_DIR / f"{clean_name}_{clean_roll}.jpg",
                                          quality=92)
                st.session_state["flash"] = f"Enrolled {clean_name.replace('_', ' ')} ({clean_roll})."
                st.rerun()
    st.caption("On Streamlit Cloud, students enrolled here are lost on restart. For permanent "
               "enrolment, commit the photo to your private GitHub repo's `student_faces/` folder.")
