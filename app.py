import os
import sys
import io
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage,
    PageBreak,
)

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.video.video_analyzer import VideoAnalyzer
from src.audio.audio_analyzer import AudioAnalyzer
from src.fusion.fusion_engine import FusionEngine


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="AViGuard",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    /* ---------- Global layout ---------- */
    .main {
        padding-top: 0.75rem;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hero ---------- */
    .av-hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem 1.75rem 2.2rem;
        border-radius: 22px;
        background:
            radial-gradient(circle at 88% 18%, rgba(99,102,241,.34), transparent 30%),
            radial-gradient(circle at 12% 100%, rgba(14,165,233,.22), transparent 32%),
            linear-gradient(135deg, #0b1220 0%, #172033 55%, #25245a 100%);
        color: #ffffff;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 14px 35px rgba(15,23,42,.16);
    }

    .av-hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .35rem .65rem;
        border-radius: 999px;
        background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.14);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
        margin-bottom: .8rem;
    }

    .av-hero h1 {
        font-size: clamp(2.25rem, 5vw, 3.35rem);
        line-height: 1.05;
        margin: 0 0 .55rem 0;
        letter-spacing: -.035em;
    }

    .av-hero p {
        max-width: 900px;
        font-size: 1.05rem;
        line-height: 1.65;
        color: rgba(255,255,255,.86);
        margin: 0;
    }

    .av-hero-chips {
        display: flex;
        flex-wrap: wrap;
        gap: .45rem;
        margin-top: 1rem;
    }

    .av-chip {
        display: inline-block;
        padding: .32rem .62rem;
        border-radius: 999px;
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.13);
        color: rgba(255,255,255,.88);
        font-size: .76rem;
        font-weight: 600;
    }

    /* ---------- Sections ---------- */
    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -.02em;
        margin-top: 1rem;
        margin-bottom: .65rem;
    }

    .section-subtitle {
        color: #64748b;
        font-size: .92rem;
        margin-top: -.25rem;
        margin-bottom: .85rem;
    }

    .info-card,
    .evidence-card {
        padding: 1rem 1.1rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        margin-bottom: .7rem;
        box-shadow: 0 5px 16px rgba(15,23,42,.045);
    }

    .info-card strong,
    .evidence-card strong {
        color: #0f172a;
    }

    .decision-card {
        padding: 1.15rem;
        border-radius: 15px;
        border: 1px solid #bfdbfe;
        background: linear-gradient(135deg, #eff6ff, #f8fbff);
        margin: .6rem 0;
        box-shadow: 0 5px 16px rgba(37,99,235,.06);
    }

    .status-good,
    .status-warning,
    .status-bad {
        padding: .78rem 1rem;
        border-radius: 12px;
        font-weight: 700;
        margin: .45rem 0;
    }

    .status-good {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
    }

    .status-warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
    }

    .status-bad {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }

    .small-note {
        color: #64748b;
        font-size: .84rem;
        line-height: 1.55;
    }

    .formula-box {
        padding: .9rem 1rem;
        border-radius: 12px;
        background: #0f172a;
        color: #e2e8f0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        margin: .75rem 0;
        border: 1px solid #1e293b;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    .av-side-brand {
        display: flex;
        align-items: center;
        gap: .7rem;
        margin-bottom: .2rem;
    }

    .av-side-logo {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        background: linear-gradient(135deg, #0f172a, #3730a3);
        color: #fff;
        font-size: 1.25rem;
        box-shadow: 0 7px 16px rgba(49,46,129,.20);
    }

    .av-side-title {
        font-size: 1.18rem;
        font-weight: 850;
        color: #0f172a;
        line-height: 1.1;
    }

    .av-side-subtitle {
        color: #64748b;
        font-size: .72rem;
        margin-top: .12rem;
    }

    .av-side-label {
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .09em;
        text-transform: uppercase;
        color: #64748b;
        margin: 1rem 0 .45rem 0;
    }

    .av-pipeline {
        display: flex;
        flex-direction: column;
        gap: 0;
        margin: .35rem 0 .75rem 0;
    }

    .av-node {
        position: relative;
        display: flex;
        align-items: center;
        gap: .65rem;
        padding: .62rem .68rem;
        border: 1px solid #dbe3ee;
        border-radius: 12px;
        background: rgba(255,255,255,.88);
        box-shadow: 0 3px 9px rgba(15,23,42,.035);
    }

    .av-node-icon {
        flex: 0 0 30px;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 9px;
        font-size: .92rem;
    }

    .av-node-main {
        min-width: 0;
        flex: 1;
    }

    .av-node-title {
        color: #0f172a;
        font-size: .78rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .av-node-meta {
        color: #64748b;
        font-size: .66rem;
        line-height: 1.35;
        margin-top: .13rem;
    }

    .av-arrow {
        text-align: center;
        color: #94a3b8;
        font-size: .72rem;
        line-height: .9;
        padding: .13rem 0;
    }

    .av-input .av-node-icon { background: #e0f2fe; color: #0369a1; }
    .av-video .av-node-icon { background: #dbeafe; color: #1d4ed8; }
    .av-audio .av-node-icon { background: #ede9fe; color: #6d28d9; }
    .av-quality .av-node-icon { background: #cffafe; color: #0e7490; }
    .av-reliability .av-node-icon { background: #ffedd5; color: #c2410c; }
    .av-fusion .av-node-icon { background: #dcfce7; color: #15803d; }
    .av-consistency .av-node-icon { background: #fce7f3; color: #be185d; }
    .av-output .av-node-icon { background: #e2e8f0; color: #334155; }

    .av-branch {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: .42rem;
    }

    .av-branch .av-node {
        min-height: 78px;
        align-items: flex-start;
    }

    .av-mini-note {
        padding: .62rem .7rem;
        border-radius: 11px;
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        color: #64748b;
        font-size: .66rem;
        line-height: 1.45;
        margin-top: .55rem;
    }

    .av-tool-card {
        padding: .68rem .75rem;
        border-radius: 11px;
        background: rgba(255,255,255,.88);
        border: 1px solid #dbe3ee;
        margin-bottom: .42rem;
    }

    .av-tool-title {
        color: #0f172a;
        font-size: .75rem;
        font-weight: 800;
    }

    .av-tool-desc {
        color: #64748b;
        font-size: .65rem;
        line-height: 1.4;
        margin-top: .12rem;
    }

    .av-side-footer {
        text-align: center;
        color: #94a3b8;
        font-size: .64rem;
        padding-top: .25rem;
    }

    /* ---------- Streamlit widget polish ---------- */
    div[data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    div[data-testid="stButton"] > button {
        border-radius: 11px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        border-radius: 13px;
        padding: .7rem .85rem;
        background: rgba(255,255,255,.72);
        box-shadow: 0 4px 12px rgba(15,23,42,.035);
    }

    /* ---------- Responsive sidebar ---------- */
    @media (max-width: 900px) {
        .av-hero {
            padding: 1.45rem 1.25rem;
        }

        .av-hero p {
            font-size: .94rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# MODEL LOADERS
# ---------------------------------------------------------

@st.cache_resource
def load_video_model():
    return VideoAnalyzer()


@st.cache_resource
def load_audio_model():
    return AudioAnalyzer()


@st.cache_resource
def load_fusion_engine():
    return FusionEngine()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "analysis_file_name" not in st.session_state:
    st.session_state.analysis_file_name = None

if "analysis_file_bytes" not in st.session_state:
    st.session_state.analysis_file_bytes = None

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value, digits=1):
    return f"{safe_float(value) * 100:.{digits}f}%"


def create_temp_video(file_bytes, filename):
    suffix = os.path.splitext(filename)[1] or ".mp4"

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    tmp.write(file_bytes)
    tmp.close()

    return tmp.name


def extract_sampled_frames(video_path, num_frames=6):
    """
    Extract evenly spaced RGB frames from the analyzed video.
    These are evidence snapshots for human inspection.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return []

    num_frames = min(num_frames, total_frames)

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int,
    )

    frames = []

    for idx in indices:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx),
        )

        success, frame = cap.read()

        if success:
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )
            frames.append(frame_rgb)

    cap.release()

    return frames


def frame_to_png_bytes(frame):
    """
    Convert an RGB numpy frame to PNG bytes.
    """

    success, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
    )

    if not success:
        return None

    return io.BytesIO(encoded.tobytes())


def build_decision_explanation(visual, audio, fusion):
    """
    Build a human-readable decision trace from the actual
    outputs returned by the three AViGuard components.
    """

    top_visual = visual.get(
        "top_prediction",
        {},
    )

    top_audio = audio.get(
        "top_prediction",
        {},
    )

    return {
        "visual_label": top_visual.get(
            "label",
            "Unknown",
        ),
        "visual_category": top_visual.get(
            "category",
            "Unknown",
        ),
        "visual_confidence": safe_float(
            top_visual.get("confidence", 0)
        ),
        "visual_reliability": safe_float(
            visual.get("reliability", 0)
        ),
        "audio_label": top_audio.get(
            "label",
            "No audio",
        ),
        "audio_category": top_audio.get(
            "category",
            "other",
        ),
        "audio_confidence": safe_float(
            top_audio.get("confidence", 0)
        ),
        "audio_reliability": safe_float(
            audio.get("reliability", 0)
        ),
        "visual_weight": safe_float(
            fusion.get("visual_weight", 0)
        ),
        "audio_weight": safe_float(
            fusion.get("audio_weight", 0)
        ),
        "agreement_score": safe_float(
            fusion.get("agreement_score", 0)
        ),
        "agreement": fusion.get(
            "agreement",
            "Unknown",
        ),
        "context": fusion.get(
            "context",
            "Unknown",
        ),
        "visual_fusion_category": fusion.get(
            "visual_category",
            top_visual.get("category", "Unknown"),
        ),
        "audio_fusion_category": fusion.get(
            "audio_category",
            top_audio.get("category", "Unknown"),
        ),
        "evidence_strength": safe_float(
            fusion.get(
                "consistency_adjusted_strength",
                fusion.get("evidence_strength", 0),
            )
        ),
    }


def render_decision_explanation(visual, audio, fusion):
    """
    Render the decision trace in the Streamlit interface.
    """

    e = build_decision_explanation(
        visual,
        audio,
        fusion,
    )

    st.markdown(
        '<div class="section-title">🧠 Why did AViGuard make this decision?</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "AViGuard first analyzes the visual and acoustic streams independently. "
        "It then estimates the reliability of each modality, uses those "
        "reliabilities for adaptive fusion, and finally checks whether the "
        "two interpretations are semantically consistent."
    )

    st.markdown("#### Step 1 — Independent visual evidence")

    st.write(
        f"🎥 The visual branch detected **{e['visual_label']}** "
        f"({e['visual_category']}) with model confidence "
        f"**{pct(e['visual_confidence'], 1)}**."
    )

    st.write(
        f"The resulting visual reliability score is "
        f"**{pct(e['visual_reliability'], 1)}**."
    )

    st.markdown("#### Step 2 — Independent acoustic evidence")

    if audio.get("has_audio", True):
        st.write(
            f"🔊 The acoustic branch detected **{e['audio_label']}** "
            f"({e['audio_category']}) with model confidence "
            f"**{pct(e['audio_confidence'], 1)}**."
        )

        st.write(
            f"The resulting audio reliability score is "
            f"**{pct(e['audio_reliability'], 1)}**."
        )
    else:
        st.warning(
            "No usable audio stream was detected. AViGuard therefore "
            "does not assign acoustic evidence a fusion contribution."
        )

    st.markdown("#### Step 3 — Reliability-aware fusion")

    st.markdown(
        f"""
        <div class="formula-box">
        Visual contribution = {pct(e['visual_weight'], 1)}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Audio contribution = {pct(e['audio_weight'], 1)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "The contribution values are derived from the estimated modality "
        "reliabilities. This allows the system to avoid treating both "
        "streams as equally trustworthy when their signal conditions differ."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Visual contribution",
            pct(e["visual_weight"], 1),
        )

    with c2:
        st.metric(
            "Audio contribution",
            pct(e["audio_weight"], 1),
        )

    st.markdown("#### Step 4 — Cross-modal consistency")

    st.write(
        f"Visual semantic category: **{e['visual_fusion_category']}**"
    )

    st.write(
        f"Audio semantic category: **{e['audio_fusion_category']}**"
    )

    st.write(
        f"Agreement score: **{pct(e['agreement_score'], 1)}**"
    )

    st.write(
        f"Relationship: **{e['agreement']}**"
    )

    st.write(
        f"System interpretation: **{e['context']}**"
    )

    if e["agreement_score"] >= 0.85:
        st.success(
            "Both modalities provide strongly consistent evidence for the "
            "same broad context."
        )
    elif e["agreement_score"] >= 0.60:
        st.warning(
            "The modalities are related, but their evidence is not identical."
        )
    else:
        st.error(
            "The modalities provide substantially different semantic evidence."
        )

    st.markdown("#### Step 5 — Final evidence strength")

    st.metric(
        "Consistency-adjusted evidence strength",
        pct(e["evidence_strength"], 1),
    )

    st.caption(
        "Evidence strength and reliability are engineering scores used for "
        "evidence aggregation; they are not calibrated probabilities."
    )


def create_waveform_figure(waveform, sample_rate=16000):
    """
    Create a matplotlib waveform figure.
    """

    waveform = np.asarray(waveform).flatten()

    if len(waveform) == 0:
        return None

    max_points = 5000

    if len(waveform) > max_points:
        indices = np.linspace(
            0,
            len(waveform) - 1,
            max_points,
        ).astype(int)

        waveform_display = waveform[indices]

        time_axis = (
            indices / max(
                safe_float(sample_rate, 16000),
                1,
            )
        )
    else:
        waveform_display = waveform

        time_axis = (
            np.arange(len(waveform))
            / max(
                safe_float(sample_rate, 16000),
                1,
            )
        )

    fig, ax = plt.subplots(
        figsize=(10, 3),
    )

    ax.plot(
        time_axis,
        waveform_display,
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Acoustic Waveform")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    return fig


def render_evidence_inspector(file_bytes, filename, audio_result):
    """
    Show sampled temporal frames and acoustic evidence.
    """

    st.markdown(
        '<div class="section-title">🎞️ Temporal Visual Evidence</div>',
        unsafe_allow_html=True,
    )

    temp_path = None

    try:
        temp_path = create_temp_video(
            file_bytes,
            filename,
        )

        frames = extract_sampled_frames(
            temp_path,
            num_frames=6,
        )

        if frames:

            st.caption(
                "Six evenly spaced frames are displayed so the user can "
                "inspect the temporal visual content used by the system."
            )

            frame_cols = st.columns(3)

            for i, frame in enumerate(frames):

                with frame_cols[i % 3]:

                    st.image(
                        frame,
                        caption=f"Sampled Frame {i + 1}",
                        use_container_width=True,
                    )

        else:
            st.warning(
                "Unable to extract visual evidence frames."
            )

    finally:

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    st.markdown(
        '<div class="section-title">🔊 Acoustic Evidence</div>',
        unsafe_allow_html=True,
    )

    waveform = audio_result.get("waveform")

    if waveform is not None and len(waveform) > 0:

        fig = create_waveform_figure(
            waveform,
            audio_result.get(
                "sample_rate",
                16000,
            ),
        )

        if fig is not None:
            st.pyplot(
                fig,
                use_container_width=True,
            )
            plt.close(fig)

    else:

        st.warning(
            "No acoustic waveform is available."
        )

    st.markdown(
        '<div class="section-title">📊 Acoustic Signal Features</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "RMS Energy",
            f"{safe_float(audio_result.get('rms', 0)):.4f}",
        )

    with c2:
        st.metric(
            "Spectral Centroid",
            f"{safe_float(audio_result.get('spectral_centroid', 0)):.1f}",
        )

    with c3:
        st.metric(
            "Zero Crossing Rate",
            f"{safe_float(audio_result.get('zero_crossing_rate', 0)):.4f}",
        )

    with c4:
        st.metric(
            "Non-Silent Ratio",
            pct(
                audio_result.get(
                    "non_silent_ratio",
                    0,
                ),
                1,
            ),
        )

    st.caption(
        "These signal-level measurements provide additional acoustic "
        "context beyond the AST classification."
    )


def make_table(data, widths):
    table = Table(
        data,
        colWidths=widths,
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1f2937"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8.5,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.whitesmoke,
                        colors.HexColor("#f3f4f6"),
                    ],
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    return table


def build_pdf_report(
    file_name,
    visual,
    audio,
    fusion,
    frame_pngs=None,
):
    """
    Generate a complete downloadable AViGuard PDF report.
    """

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=38,
        leftMargin=38,
        topMargin=38,
        bottomMargin=38,
        title="AViGuard Analysis Report",
        author="AViGuard",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=24,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=15,
        spaceBefore=12,
        spaceAfter=7,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9.2,
        leading=13,
        spaceAfter=5,
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#4b5563"),
    )

    story = []

    top_visual = visual.get("top_prediction", {})
    top_audio = audio.get("top_prediction", {})

    visual_label = top_visual.get("label", "Unknown")
    visual_category = top_visual.get("category", "Unknown")
    visual_conf = safe_float(top_visual.get("confidence", 0))
    visual_rel = safe_float(visual.get("reliability", 0))

    audio_label = top_audio.get("label", "No audio")
    audio_category = top_audio.get("category", "other")
    audio_conf = safe_float(top_audio.get("confidence", 0))
    audio_rel = safe_float(audio.get("reliability", 0))

    visual_weight = safe_float(
        fusion.get("visual_weight", 0)
    )

    audio_weight = safe_float(
        fusion.get("audio_weight", 0)
    )

    agreement_score = safe_float(
        fusion.get("agreement_score", 0)
    )

    evidence_strength = safe_float(
        fusion.get(
            "consistency_adjusted_strength",
            fusion.get("evidence_strength", 0),
        )
    )

    agreement = fusion.get(
        "agreement",
        "Unknown",
    )

    context = fusion.get(
        "context",
        "Unknown",
    )

    # ---------------------------------------------------------
    # COVER
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "AViGuard",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Context-Aware Audio-Visual Event Detection & "
            "Cross-Modal Consistency Analysis",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Analyzed media:</b> {file_name}",
            body_style,
        )
    )

    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "1. Multimodal Decision Summary",
            heading_style,
        )
    )

    summary_data = [
        ["Component", "Result"],
        [
            "Visual interpretation",
            f"{visual_label} ({visual_category})",
        ],
        [
            "Visual model confidence",
            pct(visual_conf, 2),
        ],
        [
            "Visual reliability",
            pct(visual_rel, 2),
        ],
        [
            "Audio interpretation",
            f"{audio_label} ({audio_category})",
        ],
        [
            "Audio model confidence",
            pct(audio_conf, 2),
        ],
        [
            "Audio reliability",
            pct(audio_rel, 2),
        ],
        [
            "Cross-modal agreement",
            pct(agreement_score, 2),
        ],
        [
            "Relationship",
            str(agreement),
        ],
        [
            "Evidence strength",
            pct(evidence_strength, 2),
        ],
    ]

    story.append(
        make_table(
            summary_data,
            [2.45 * inch, 4.0 * inch],
        )
    )

    # ---------------------------------------------------------
    # WHY
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "2. Why AViGuard Reached This Decision",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            "AViGuard treats the visual and acoustic streams as "
            "independent evidence sources. Each branch produces a "
            "classification and a reliability estimate. The fusion "
            "stage then combines the evidence using the modality "
            "reliabilities and evaluates semantic consistency.",
            body_style,
        )
    )

    explanation_rows = [
        ["Decision stage", "Observed evidence"],
        [
            "Visual evidence",
            f"The visual branch identified "
            f"<b>{visual_label}</b> with "
            f"{pct(visual_conf, 1)} model confidence.",
        ],
        [
            "Visual reliability",
            f"Estimated visual reliability: "
            f"<b>{pct(visual_rel, 1)}</b>.",
        ],
        [
            "Acoustic evidence",
            f"The acoustic branch identified "
            f"<b>{audio_label}</b> with "
            f"{pct(audio_conf, 1)} model confidence.",
        ],
        [
            "Audio reliability",
            f"Estimated audio reliability: "
            f"<b>{pct(audio_rel, 1)}</b>.",
        ],
        [
            "Adaptive fusion",
            f"Visual contribution: <b>{pct(visual_weight, 1)}</b>; "
            f"audio contribution: <b>{pct(audio_weight, 1)}</b>.",
        ],
        [
            "Consistency",
            f"Agreement score: <b>{pct(agreement_score, 1)}</b>; "
            f"relationship: <b>{agreement}</b>.",
        ],
        [
            "System interpretation",
            str(context),
        ],
    ]

    story.append(
        make_table(
            explanation_rows,
            [1.65 * inch, 4.8 * inch],
        )
    )

    story.append(
        Spacer(1, 7)
    )

    story.append(
        Paragraph(
            "Note: reliability, agreement and evidence-strength values "
            "are engineering scores used by AViGuard for evidence "
            "aggregation. They should not be interpreted as calibrated "
            "probabilities.",
            small_style,
        )
    )

    # ---------------------------------------------------------
    # VISUAL
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "3. Visual Analysis",
            heading_style,
        )
    )

    visual_rows = [
        ["Metric", "Value"],
        [
            "Top prediction",
            visual_label,
        ],
        [
            "Semantic category",
            visual_category,
        ],
        [
            "Model confidence",
            pct(visual_conf, 2),
        ],
        [
            "Visual reliability",
            pct(visual_rel, 2),
        ],
        [
            "Mean brightness",
            f"{safe_float(visual.get('brightness', visual.get('mean_brightness', 0))):.4f}",
        ],
        [
            "Sharpness",
            f"{safe_float(visual.get('sharpness', 0)):.2f}",
        ],
        [
            "Device",
            str(visual.get("device", "CPU")),
        ],
    ]

    story.append(
        make_table(
            visual_rows,
            [2.45 * inch, 4.0 * inch],
        )
    )

    story.append(
        Paragraph(
            "Top visual predictions",
            heading_style,
        )
    )

    visual_prediction_rows = [
        ["Rank", "Event", "Category", "Confidence"],
    ]

    for i, prediction in enumerate(
        visual.get("predictions", [])[:5],
        start=1,
    ):
        visual_prediction_rows.append(
            [
                str(i),
                str(prediction.get("label", "Unknown")),
                str(prediction.get("category", "Unknown")),
                pct(
                    prediction.get("confidence", 0),
                    2,
                ),
            ]
        )

    if len(visual_prediction_rows) > 1:
        story.append(
            make_table(
                visual_prediction_rows,
                [
                    0.55 * inch,
                    2.8 * inch,
                    1.55 * inch,
                    1.55 * inch,
                ],
            )
        )

    # ---------------------------------------------------------
    # AUDIO
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "4. Acoustic Analysis",
            heading_style,
        )
    )

    audio_rows = [
        ["Metric", "Value"],
        [
            "Audio available",
            str(audio.get("has_audio", True)),
        ],
        [
            "Top prediction",
            audio_label,
        ],
        [
            "Semantic category",
            audio_category,
        ],
        [
            "Model confidence",
            pct(audio_conf, 2),
        ],
        [
            "Audio reliability",
            pct(audio_rel, 2),
        ],
        [
            "Sample rate",
            str(audio.get("sample_rate", "N/A")),
        ],
        [
            "Duration",
            str(audio.get("duration", "N/A")),
        ],
        [
            "RMS energy",
            f"{safe_float(audio.get('rms', 0)):.6f}",
        ],
        [
            "Spectral centroid",
            f"{safe_float(audio.get('spectral_centroid', 0)):.2f}",
        ],
        [
            "Zero crossing rate",
            f"{safe_float(audio.get('zero_crossing_rate', 0)):.6f}",
        ],
        [
            "Non-silent ratio",
            pct(audio.get("non_silent_ratio", 0), 2),
        ],
        [
            "Device",
            str(audio.get("device", "CPU")),
        ],
    ]

    story.append(
        make_table(
            audio_rows,
            [2.45 * inch, 4.0 * inch],
        )
    )

    story.append(
        Paragraph(
            "Top acoustic predictions",
            heading_style,
        )
    )

    audio_prediction_rows = [
        ["Rank", "Sound", "Category", "Confidence"],
    ]

    for i, prediction in enumerate(
        audio.get("predictions", [])[:5],
        start=1,
    ):
        audio_prediction_rows.append(
            [
                str(i),
                str(prediction.get("label", "Unknown")),
                str(prediction.get("category", "Unknown")),
                pct(
                    prediction.get("confidence", 0),
                    2,
                ),
            ]
        )

    if len(audio_prediction_rows) > 1:
        story.append(
            make_table(
                audio_prediction_rows,
                [
                    0.55 * inch,
                    2.8 * inch,
                    1.55 * inch,
                    1.55 * inch,
                ],
            )
        )

    # ---------------------------------------------------------
    # WAVEFORM
    # ---------------------------------------------------------

    waveform = audio.get("waveform")

    if waveform is not None and len(waveform) > 0:

        fig = create_waveform_figure(
            waveform,
            audio.get("sample_rate", 16000),
        )

        if fig is not None:

            image_buffer = io.BytesIO()

            fig.savefig(
                image_buffer,
                format="png",
                dpi=140,
                bbox_inches="tight",
            )

            plt.close(fig)

            image_buffer.seek(0)

            story.append(
                Paragraph(
                    "Acoustic Waveform",
                    heading_style,
                )
            )

            story.append(
                ReportLabImage(
                    image_buffer,
                    width=6.6 * inch,
                    height=2.0 * inch,
                )
            )

    # ---------------------------------------------------------
    # FUSION
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "5. Reliability-Aware Fusion",
            heading_style,
        )
    )

    fusion_rows = [
        ["Fusion metric", "Value"],
        [
            "Visual contribution",
            pct(visual_weight, 2),
        ],
        [
            "Audio contribution",
            pct(audio_weight, 2),
        ],
        [
            "Agreement score",
            pct(agreement_score, 2),
        ],
        [
            "Relationship",
            str(agreement),
        ],
        [
            "Visual semantic category",
            str(
                fusion.get(
                    "visual_category",
                    visual_category,
                )
            ),
        ],
        [
            "Audio semantic category",
            str(
                fusion.get(
                    "audio_category",
                    audio_category,
                )
            ),
        ],
        [
            "System context",
            str(context),
        ],
        [
            "Consistency-adjusted evidence strength",
            pct(evidence_strength, 2),
        ],
    ]

    story.append(
        make_table(
            fusion_rows,
            [3.0 * inch, 3.45 * inch],
        )
    )

    # ---------------------------------------------------------
    # TEMPORAL FRAMES
    # ---------------------------------------------------------

    if frame_pngs:

        story.append(PageBreak())

        story.append(
            Paragraph(
                "6. Sampled Temporal Visual Evidence",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                "The following frames were sampled at approximately "
                "evenly spaced positions across the video. They are "
                "provided for human inspection of the visual evidence.",
                body_style,
            )
        )

        for i, frame_buffer in enumerate(
            frame_pngs[:6],
            start=1,
        ):

            try:

                frame_buffer.seek(0)

                story.append(
                    Paragraph(
                        f"Sampled Frame {i}",
                        body_style,
                    )
                )

                story.append(
                    ReportLabImage(
                        frame_buffer,
                        width=3.15 * inch,
                        height=2.1 * inch,
                    )
                )

                story.append(
                    Spacer(1, 8)
                )

            except Exception:
                continue

    # ---------------------------------------------------------
    # ARCHITECTURE
    # ---------------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "7. AViGuard System Architecture",
            heading_style,
        )
    )

    architecture_text = """
    <b>Video branch:</b> R3D-18 pretrained on Kinetics-400. The video
    pipeline samples temporal frames, performs spatial preprocessing,
    and extracts visual-temporal event evidence.

    <br/><br/>

    <b>Audio branch:</b> Audio Spectrogram Transformer (AST) pretrained
    on AudioSet. The audio pipeline extracts a 16 kHz mono waveform and
    processes acoustic information for environmental sound recognition.

    <br/><br/>

    <b>Signal quality:</b> Visual quality is characterized using
    measurements such as brightness and sharpness. Acoustic quality uses
    signal-level measurements such as RMS energy and non-silent ratio.

    <br/><br/>

    <b>Reliability-aware fusion:</b> The independent modality outputs
    are combined using estimated modality reliability instead of assuming
    that the visual and acoustic streams are always equally trustworthy.

    <br/><br/>

    <b>Cross-modal consistency:</b> Model predictions are mapped into
    broad semantic categories and compared to determine whether the two
    evidence streams support a similar context.
    """

    story.append(
        Paragraph(
            architecture_text,
            body_style,
        )
    )

    story.append(
        Paragraph(
            "Engineering interpretation",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            "AViGuard is a multimodal inference and evidence-fusion "
            "prototype. The pretrained perception models provide the "
            "base visual and acoustic predictions, while the quality "
            "assessment, reliability estimation, semantic consistency "
            "analysis, reporting and user interface form the custom "
            "system layer.",
            body_style,
        )
    )

    story.append(
        Paragraph(
            "Important note: AViGuard does not claim that its reliability "
            "or evidence-strength values are calibrated probabilities, "
            "and the system should not be interpreted as a trained "
            "end-to-end multimodal model.",
            small_style,
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ---------------------------------------------------------
# SIDEBAR — AViGuard SYSTEM MAP
# ---------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="av-side-brand">
            <div class="av-side-logo">◈</div>
            <div>
                <div class="av-side-title">AViGuard</div>
                <div class="av-side-subtitle">Multimodal evidence intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="av-side-label">System architecture</div>',
        unsafe_allow_html=True,
    )

    # Use st.html rather than st.markdown for the architecture diagram.
    # st.markdown can interpret indented HTML as a code block; st.html
    # renders the diagram as actual HTML/CSS in Streamlit.
    st.html(
        """
        <style>
        .aviguard-map {
            width: 100%;
            box-sizing: border-box;
            font-family: inherit;
            color: #0f172a;
        }

        .aviguard-map * {
            box-sizing: border-box;
        }

        .av-map-node {
            width: 100%;
            display: flex;
            align-items: center;
            gap: 9px;
            padding: 9px 10px;
            border: 1px solid #dbe3ee;
            border-radius: 12px;
            background: #ffffff;
            box-shadow: 0 2px 7px rgba(15, 23, 42, 0.045);
        }

        .av-map-icon {
            width: 30px;
            height: 30px;
            flex: 0 0 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 9px;
            font-size: 14px;
            line-height: 1;
        }

        .av-map-content {
            min-width: 0;
            flex: 1;
        }

        .av-map-title {
            font-size: 12px;
            line-height: 1.2;
            font-weight: 800;
            letter-spacing: -0.01em;
            color: #0f172a;
        }

        .av-map-meta {
            margin-top: 2px;
            font-size: 10px;
            line-height: 1.35;
            color: #64748b;
        }

        .av-map-arrow {
            height: 17px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #94a3b8;
            font-size: 13px;
            line-height: 1;
        }

        .av-map-branches {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 6px;
        }

        .av-map-branches .av-map-node {
            align-items: flex-start;
            min-height: 72px;
            padding: 8px;
        }

        .av-map-branches .av-map-icon {
            width: 27px;
            height: 27px;
            flex-basis: 27px;
            font-size: 13px;
        }

        .av-map-branches .av-map-title {
            font-size: 11px;
        }

        .av-map-branches .av-map-meta {
            font-size: 9px;
        }

        .av-map-video .av-map-icon {
            background: #dbeafe;
            color: #1d4ed8;
        }

        .av-map-audio .av-map-icon {
            background: #ede9fe;
            color: #6d28d9;
        }

        .av-map-input .av-map-icon {
            background: #e0f2fe;
            color: #0369a1;
        }

        .av-map-quality .av-map-icon {
            background: #cffafe;
            color: #0e7490;
        }

        .av-map-reliability .av-map-icon {
            background: #ffedd5;
            color: #c2410c;
        }

        .av-map-fusion .av-map-icon {
            background: #dcfce7;
            color: #15803d;
        }

        .av-map-consistency .av-map-icon {
            background: #fce7f3;
            color: #be185d;
        }

        .av-map-output .av-map-icon {
            background: #e2e8f0;
            color: #334155;
        }

        .av-map-mini {
            margin-top: 8px;
            padding: 8px 9px;
            border: 1px dashed #cbd5e1;
            border-radius: 10px;
            background: #f8fafc;
            color: #64748b;
            font-size: 9px;
            line-height: 1.45;
        }

        .av-map-mini strong {
            color: #334155;
            font-weight: 800;
        }

        @media (max-width: 340px) {
            .av-map-branches {
                grid-template-columns: 1fr;
            }

            .av-map-branches .av-map-node {
                min-height: auto;
            }
        }
        </style>

        <div class="aviguard-map" aria-label="AViGuard system architecture">

            <div class="av-map-node av-map-input">
                <div class="av-map-icon">📥</div>
                <div class="av-map-content">
                    <div class="av-map-title">Media Input</div>
                    <div class="av-map-meta">Video + optional audio stream</div>
                </div>
            </div>

            <div class="av-map-arrow" aria-hidden="true">↓</div>

            <div class="av-map-branches">

                <div class="av-map-node av-map-video">
                    <div class="av-map-icon">🎥</div>
                    <div class="av-map-content">
                        <div class="av-map-title">Visual</div>
                        <div class="av-map-meta">R3D-18<br>Kinetics-400</div>
                    </div>
                </div>

                <div class="av-map-node av-map-audio">
                    <div class="av-map-icon">🔊</div>
                    <div class="av-map-content">
                        <div class="av-map-title">Acoustic</div>
                        <div class="av-map-meta">AST<br>AudioSet • 16 kHz</div>
                    </div>
                </div>

            </div>

            <div class="av-map-arrow" aria-hidden="true">↓</div>

            <div class="av-map-node av-map-quality">
                <div class="av-map-icon">📊</div>
                <div class="av-map-content">
                    <div class="av-map-title">Signal Quality</div>
                    <div class="av-map-meta">Brightness • sharpness • RMS • activity</div>
                </div>
            </div>

            <div class="av-map-arrow" aria-hidden="true">↓</div>

            <div class="av-map-node av-map-reliability">
                <div class="av-map-icon">⚖️</div>
                <div class="av-map-content">
                    <div class="av-map-title">Modality Reliability</div>
                    <div class="av-map-meta">Confidence + quality → adaptive trust</div>
                </div>
            </div>

            <div class="av-map-arrow" aria-hidden="true">↓</div>

            <div class="av-map-node av-map-fusion">
                <div class="av-map-icon">🔗</div>
                <div class="av-map-content">
                    <div class="av-map-title">Reliability-Aware Fusion</div>
                    <div class="av-map-meta">Adaptive evidence contribution</div>
                </div>
            </div>

            <div class="av-map-arrow" aria-hidden="true">↓</div>

            <div class="av-map-node av-map-consistency">
                <div class="av-map-icon">🧩</div>
                <div class="av-map-content">
                    <div class="av-map-title">Cross-Modal Consistency</div>
                    <div class="av-map-meta">Semantic agreement analysis</div>
                </div>
            </div>

            <div class="av-map-arrow" aria-hidden="true">↓</div>

            <div class="av-map-node av-map-output">
                <div class="av-map-icon">📋</div>
                <div class="av-map-content">
                    <div class="av-map-title">Interpretable Output</div>
                    <div class="av-map-meta">Evidence • diagnostics • decision trace • PDF</div>
                </div>
            </div>

            <div class="av-map-mini">
                <strong>Core principle:</strong>
                independent visual and acoustic evidence is assessed first,
                then weighted by estimated reliability before aggregation.
            </div>

        </div>
        """
    )

    st.markdown(
        '<div class="av-side-label">Analysis tools</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="av-tool-card">
            <div class="av-tool-title">🧠 Explain Decision</div>
            <div class="av-tool-desc">Trace how model evidence becomes the system interpretation.</div>
        </div>

        <div class="av-tool-card">
            <div class="av-tool-title">🎞️ Evidence Inspector</div>
            <div class="av-tool-desc">Inspect sampled frames and the acoustic waveform.</div>
        </div>

        <div class="av-tool-card">
            <div class="av-tool-title">📄 Technical Report</div>
            <div class="av-tool-desc">Generate a downloadable PDF analysis report.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="av-side-footer">
            AViGuard • Audio + Vision • Evidence Fusion
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown(
    """
    <div class="av-hero">
        <div class="av-hero-kicker">◈ Multimodal AI • Evidence Fusion</div>
        <h1>AViGuard</h1>
        <p>
            Context-Aware Audio-Visual Event Detection &amp;
            Cross-Modal Consistency Analysis
        </p>
        <div class="av-hero-chips">
            <span class="av-chip">🎥 Temporal Vision</span>
            <span class="av-chip">🔊 Acoustic Intelligence</span>
            <span class="av-chip">⚖️ Reliability-Aware Fusion</span>
            <span class="av-chip">🧩 Consistency Analysis</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    'Upload a short video to inspect visual and acoustic evidence, '
    'estimate modality reliability, and understand how the final '
    'multimodal interpretation was formed.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📤 Upload Media</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="Upload a short video for multimodal analysis.",
)


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------

if uploaded_file is not None:

    # -----------------------------------------------------
    # VIDEO PREVIEW
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🎬 Video Preview</div>',
        unsafe_allow_html=True,
    )

    st.video(uploaded_file)

    st.caption(
        f"File: {uploaded_file.name}  •  "
        f"Size: {uploaded_file.size / (1024 * 1024):.2f} MB"
    )

    # Reset old analysis if a different file was uploaded.
    current_signature = (
        uploaded_file.name,
        uploaded_file.size,
    )

    previous_signature = st.session_state.get(
        "analysis_signature"
    )

    if previous_signature != current_signature:

        st.session_state.analysis = None
        st.session_state.analysis_file_name = None
        st.session_state.analysis_file_bytes = None
        st.session_state.pdf_bytes = None
        st.session_state.analysis_signature = current_signature

    st.divider()

    analyze = st.button(
        "🚀 Analyze Video",
        type="primary",
        use_container_width=True,
    )

    if analyze:

        temp_path = None

        try:

            # -------------------------------------------------
            # SAVE UPLOAD
            # -------------------------------------------------

            file_bytes = uploaded_file.getvalue()

            st.session_state.analysis_file_bytes = file_bytes
            st.session_state.analysis_file_name = uploaded_file.name
            st.session_state.pdf_bytes = None

            suffix = os.path.splitext(
                uploaded_file.name
            )[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp:

                tmp.write(file_bytes)
                temp_path = tmp.name

            # -------------------------------------------------
            # LOAD MODELS
            # -------------------------------------------------

            with st.spinner(
                "Loading multimodal AI models..."
            ):

                video_model = load_video_model()
                audio_model = load_audio_model()
                fusion_engine = load_fusion_engine()

            # -------------------------------------------------
            # ANALYSIS
            # -------------------------------------------------

            progress = st.progress(0)

            st.info(
                "🎥 Extracting visual-temporal evidence..."
            )

            visual_result = video_model.analyze(
                temp_path
            )

            progress.progress(35)

            st.info(
                "🔊 Extracting acoustic evidence..."
            )

            audio_result = audio_model.analyze(
                temp_path
            )

            progress.progress(70)

            st.info(
                "🔗 Performing cross-modal fusion..."
            )

            fusion_result = fusion_engine.fuse(
                visual_result,
                audio_result,
            )

            progress.progress(100)

            st.success(
                "Analysis completed successfully!"
            )

            # -------------------------------------------------
            # STORE ANALYSIS
            # -------------------------------------------------

            st.session_state.analysis = {
                "visual": visual_result,
                "audio": audio_result,
                "fusion": fusion_result,
            }

        except Exception as e:

            st.error(
                "❌ Analysis failed."
            )

            st.exception(e)

        finally:

            if temp_path is not None and os.path.exists(
                temp_path
            ):

                try:
                    os.remove(temp_path)
                except OSError:
                    pass


    # ---------------------------------------------------------
    # DISPLAY STORED ANALYSIS
    # ---------------------------------------------------------

    analysis = st.session_state.get(
        "analysis"
    )

    if analysis is not None:

        visual_result = analysis["visual"]
        audio_result = analysis["audio"]
        fusion_result = analysis["fusion"]

        # -----------------------------------------------------
        # TOP SUMMARY
        # -----------------------------------------------------

        st.markdown(
            '<div class="section-title">🧠 Multimodal Summary</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Visual Evidence",
                pct(
                    visual_result["top_prediction"]["confidence"],
                    1,
                ),
            )

        with col2:
            st.metric(
                "Audio Evidence",
                pct(
                    audio_result["top_prediction"]["confidence"],
                    1,
                ),
            )

        with col3:
            st.metric(
                "Cross-Modal Agreement",
                pct(
                    fusion_result["agreement_score"],
                    0,
                ),
            )

        with col4:
            st.metric(
                "Evidence Strength",
                pct(
                    fusion_result[
                        "consistency_adjusted_strength"
                    ],
                    1,
                ),
            )

        # -----------------------------------------------------
        # TWO MODALITIES
        # -----------------------------------------------------

        st.markdown(
            '<div class="section-title">🎥 Visual & 🔊 Audio Evidence</div>',
            unsafe_allow_html=True,
        )

        visual_col, audio_col = st.columns(2)

        # -----------------------------------------------------
        # VISUAL
        # -----------------------------------------------------

        with visual_col:

            st.subheader("🎥 Visual Analysis")

            top_visual = visual_result[
                "top_prediction"
            ]

            st.markdown(
                f"""
                <div class="info-card">
                    <b>Detected event</b><br>
                    {top_visual['label']}
                    <br><br>
                    <b>Category</b><br>
                    {top_visual['category']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.metric(
                "Model Confidence",
                pct(
                    top_visual["confidence"],
                    2,
                ),
            )

            st.metric(
                "Visual Reliability",
                pct(
                    visual_result["reliability"],
                    2,
                ),
            )

            st.markdown(
                "**Top visual predictions**"
            )

            visual_rows = []

            for prediction in visual_result[
                "predictions"
            ]:

                visual_rows.append(
                    {
                        "Event": prediction["label"],
                        "Category": prediction["category"],
                        "Confidence": pct(
                            prediction["confidence"],
                            2,
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(
                    visual_rows
                ),
                use_container_width=True,
                hide_index=True,
            )

        # -----------------------------------------------------
        # AUDIO
        # -----------------------------------------------------

        with audio_col:

            st.subheader("🔊 Audio Analysis")

            top_audio = audio_result[
                "top_prediction"
            ]

            st.markdown(
                f"""
                <div class="info-card">
                    <b>Detected sound</b><br>
                    {top_audio['label']}
                    <br><br>
                    <b>Category</b><br>
                    {top_audio['category']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.metric(
                "Model Confidence",
                pct(
                    top_audio["confidence"],
                    2,
                ),
            )

            st.metric(
                "Audio Reliability",
                pct(
                    audio_result["reliability"],
                    2,
                ),
            )

            st.markdown(
                "**Top audio predictions**"
            )

            audio_rows = []

            for prediction in audio_result[
                "predictions"
            ]:

                audio_rows.append(
                    {
                        "Sound": prediction["label"],
                        "Category": prediction["category"],
                        "Confidence": pct(
                            prediction["confidence"],
                            2,
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(
                    audio_rows
                ),
                use_container_width=True,
                hide_index=True,
            )

        # -----------------------------------------------------
        # RELIABILITY FUSION
        # -----------------------------------------------------

        st.markdown(
            '<div class="section-title">⚖️ Reliability-Aware Fusion</div>',
            unsafe_allow_html=True,
        )

        fusion_col1, fusion_col2 = st.columns(2)

        with fusion_col1:

            st.markdown(
                "**Visual contribution**"
            )

            st.progress(
                min(
                    max(
                        safe_float(
                            fusion_result[
                                "visual_weight"
                            ]
                        ),
                        0.0,
                    ),
                    1.0,
                )
            )

            st.write(
                pct(
                    fusion_result[
                        "visual_weight"
                    ],
                    1,
                )
            )

        with fusion_col2:

            st.markdown(
                "**Audio contribution**"
            )

            st.progress(
                min(
                    max(
                        safe_float(
                            fusion_result[
                                "audio_weight"
                            ]
                        ),
                        0.0,
                    ),
                    1.0,
                )
            )

            st.write(
                pct(
                    fusion_result[
                        "audio_weight"
                    ],
                    1,
                )
            )

        st.caption(
            "Fusion weights are determined from estimated modality reliability."
        )

        # -----------------------------------------------------
        # CROSS MODAL CONSISTENCY
        # -----------------------------------------------------

        st.markdown(
            '<div class="section-title">🔗 Cross-Modal Consistency</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="info-card">

            <b>Visual interpretation:</b>
            {fusion_result['visual_label']}

            <br><br>

            <b>Audio interpretation:</b>
            {fusion_result['audio_label']}

            <br><br>

            <b>Relationship:</b>
            {fusion_result['agreement']}

            <br><br>

            <b>System interpretation:</b>
            {fusion_result['context']}

            </div>
            """,
            unsafe_allow_html=True,
        )

        agreement_score = safe_float(
            fusion_result["agreement_score"]
        )

        if agreement_score >= 0.85:

            st.markdown(
                """
                <div class="status-good">
                🟢 Strong cross-modal agreement
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif agreement_score >= 0.60:

            st.markdown(
                """
                <div class="status-warning">
                🟡 Related but non-identical evidence
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="status-bad">
                🔴 Potential cross-modal disagreement
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -----------------------------------------------------
        # AUDIO SIGNAL
        # -----------------------------------------------------

        st.markdown(
            '<div class="section-title">🌊 Acoustic Signal</div>',
            unsafe_allow_html=True,
        )

        waveform = audio_result.get(
            "waveform"
        )

        if waveform is not None and len(waveform) > 0:

            fig = create_waveform_figure(
                waveform,
                audio_result.get(
                    "sample_rate",
                    16000,
                ),
            )

            if fig is not None:

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

        else:

            st.warning(
                "No acoustic waveform is available."
            )

        # -----------------------------------------------------
        # QUALITY DIAGNOSTICS
        # -----------------------------------------------------

        st.markdown(
            '<div class="section-title">🔬 Signal Quality Diagnostics</div>',
            unsafe_allow_html=True,
        )

        quality_col1, quality_col2 = st.columns(2)

        with quality_col1:

            st.subheader("🎥 Video Quality")

            st.metric(
                "Brightness",
                f"{safe_float(visual_result.get('brightness', visual_result.get('mean_brightness', 0))):.3f}",
            )

            st.metric(
                "Sharpness",
                f"{safe_float(visual_result.get('sharpness', 0)):.2f}",
            )

        with quality_col2:

            st.subheader("🔊 Audio Quality")

            st.metric(
                "RMS Energy",
                f"{safe_float(audio_result.get('rms', 0)):.4f}",
            )

            st.metric(
                "Non-Silent Ratio",
                pct(
                    audio_result.get(
                        "non_silent_ratio",
                        0,
                    ),
                    1,
                ),
            )

        # -----------------------------------------------------
        # NEW: EVIDENCE & DECISION CONTROLS
        # -----------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">🔎 Evidence & Decision Analysis</div>',
            unsafe_allow_html=True,
        )

        tool_col1, tool_col2 = st.columns(2)

        with tool_col1:

            show_explanation = st.button(
                "🧠 Explain This Decision",
                use_container_width=True,
                key="explain_decision",
            )

        with tool_col2:

            show_evidence = st.button(
                "🎞️ Inspect Video + Audio Evidence",
                use_container_width=True,
                key="inspect_evidence",
            )

        # -----------------------------------------------------
        # EXPLANATION
        # -----------------------------------------------------

        if show_explanation:

            render_decision_explanation(
                visual_result,
                audio_result,
                fusion_result,
            )

        # -----------------------------------------------------
        # EVIDENCE INSPECTOR
        # -----------------------------------------------------

        if show_evidence:

            file_bytes = st.session_state.get(
                "analysis_file_bytes"
            )

            file_name = st.session_state.get(
                "analysis_file_name"
            )

            if file_bytes and file_name:

                render_evidence_inspector(
                    file_bytes,
                    file_name,
                    audio_result,
                )

            else:

                st.warning(
                    "The original uploaded media is no longer available "
                    "for evidence inspection. Please analyze the video again."
                )

        # -----------------------------------------------------
        # PDF REPORT
        # -----------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">📄 Analysis Report</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            Generate a complete technical report containing the multimodal
            decision, model predictions, reliability values, signal quality,
            fusion analysis, acoustic waveform and sampled visual frames.
            """,
        )

        generate_pdf = st.button(
            "📥 Generate Full PDF Report",
            use_container_width=True,
            key="generate_pdf",
        )

        if generate_pdf:

            file_bytes = st.session_state.get(
                "analysis_file_bytes"
            )

            file_name = st.session_state.get(
                "analysis_file_name"
            )

            if file_bytes and file_name:

                with st.spinner(
                    "Generating complete AViGuard PDF report..."
                ):

                    temp_path = None

                    try:

                        temp_path = create_temp_video(
                            file_bytes,
                            file_name,
                        )

                        frames = extract_sampled_frames(
                            temp_path,
                            num_frames=6,
                        )

                        frame_pngs = []

                        for frame in frames:

                            png = frame_to_png_bytes(
                                frame
                            )

                            if png is not None:
                                frame_pngs.append(
                                    png
                                )

                        pdf_bytes = build_pdf_report(
                            file_name=file_name,
                            visual=visual_result,
                            audio=audio_result,
                            fusion=fusion_result,
                            frame_pngs=frame_pngs,
                        )

                        st.session_state.pdf_bytes = pdf_bytes

                    except Exception as e:

                        st.error(
                            "Could not generate the PDF report."
                        )

                        st.exception(e)

                    finally:

                        if temp_path and os.path.exists(
                            temp_path
                        ):

                            try:
                                os.remove(temp_path)
                            except OSError:
                                pass

            else:

                st.warning(
                    "Please analyze the uploaded video again before "
                    "generating the report."
                )

        # -----------------------------------------------------
        # DOWNLOAD BUTTON
        # -----------------------------------------------------

        pdf_bytes = st.session_state.get(
            "pdf_bytes"
        )

        if pdf_bytes:

            st.success(
                "PDF report generated successfully!"
            )

            st.download_button(
                label="⬇️ Download AViGuard PDF Report",
                data=pdf_bytes,
                file_name="AViGuard_Analysis_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf",
            )

        # -----------------------------------------------------
        # TECHNICAL DETAILS
        # -----------------------------------------------------

        with st.expander(
            "🛠️ Technical Details"
        ):

            st.write(
                {
                    "Video model": "R3D-18 / Kinetics-400",
                    "Audio model": "AST / AudioSet",
                    "Video device": str(
                        visual_result.get(
                            "device",
                            "CPU",
                        )
                    ),
                    "Audio device": str(
                        audio_result.get(
                            "device",
                            "CPU",
                        )
                    ),
                    "Visual category": fusion_result[
                        "visual_category"
                    ],
                    "Audio category": fusion_result[
                        "audio_category"
                    ],
                    "Agreement": fusion_result[
                        "agreement"
                    ],
                    "Visual weight": fusion_result[
                        "visual_weight"
                    ],
                    "Audio weight": fusion_result[
                        "audio_weight"
                    ],
                    "Audio stream available": audio_result.get(
                        "has_audio",
                        True,
                    ),
                }
            )


else:

    # ---------------------------------------------------------
    # EMPTY STATE
    # ---------------------------------------------------------

    st.markdown(
        """
        <div class="info-card">

        <h3>🎬 Ready for multimodal analysis</h3>

        Upload an MP4 or other supported video format above.

        <br><br>

        AViGuard will analyze the media through two independent
        perception pipelines and then compare their evidence.

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "🎥 **Visual Intelligence**\n\n"
            "Temporal event recognition using R3D-18."
        )

    with col2:

        st.info(
            "🔊 **Audio Intelligence**\n\n"
            "Environmental sound recognition using AST."
        )

    with col3:

        st.info(
            "🔗 **Consistency Analysis**\n\n"
            "Reliability-aware cross-modal evidence fusion."
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "AViGuard • Designed & Developed by V T Rushi Kannan"
)
