import io
import os
import tempfile

import numpy as np
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
    Image,
    PageBreak,
)


def _safe(value, default="N/A"):
    if value is None:
        return default

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def _percent(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return _safe(value)


def create_waveform_image(waveform, sample_rate=16000):
    if waveform is None:
        return None

    try:
        waveform = np.asarray(waveform).flatten()

        if len(waveform) == 0:
            return None

        # Limit report size
        max_samples = sample_rate * 30
        waveform = waveform[:max_samples]

        time = np.arange(len(waveform)) / sample_rate

        fig = plt.figure(figsize=(8, 2.5))
        plt.plot(time, waveform)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title("Acoustic Waveform")
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150)
        plt.close(fig)

        buffer.seek(0)
        return buffer

    except Exception:
        return None


def create_pdf_report(
    video_path,
    visual,
    audio,
    fusion,
    frame_images=None,
):
    """
    Create a complete AViGuard PDF analysis report.

    Returns:
        bytes
    """

    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "AViGuardTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=24,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "AViGuardSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=11,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        "AViGuardHeading",
        parent=styles["Heading2"],
        fontSize=16,
        spaceBefore=14,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "AViGuardBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=6,
    )

    story = []

    # =========================================================
    # TITLE
    # =========================================================

    story.append(Paragraph("AViGuard", title_style))

    story.append(
        Paragraph(
            "Context-Aware Audio-Visual Event Detection & "
            "Cross-Modal Consistency Analysis",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Analyzed media:</b> {os.path.basename(video_path)}",
            body_style,
        )
    )

    story.append(Spacer(1, 10))

    # =========================================================
    # EXECUTIVE SUMMARY
    # =========================================================

    story.append(
        Paragraph("1. Multimodal Decision Summary", heading_style)
    )

    visual_label = visual.get("top_prediction", "Unknown")
    audio_label = audio.get("top_prediction", "No audio")
    relationship = fusion.get(
        "agreement_text",
        fusion.get("relationship", "Unknown"),
    )

    evidence_strength = fusion.get(
        "evidence_strength",
        fusion.get("evidence", None),
    )

    summary_data = [
        ["Component", "Result"],
        [
            "Visual interpretation",
            _safe(visual_label),
        ],
        [
            "Visual confidence",
            _percent(
                visual.get("confidence", visual.get("top_confidence", 0))
            ),
        ],
        [
            "Visual reliability",
            _percent(visual.get("reliability", 0)),
        ],
        [
            "Audio interpretation",
            _safe(audio_label),
        ],
        [
            "Audio confidence",
            _percent(
                audio.get("confidence", audio.get("top_confidence", 0))
            ),
        ],
        [
            "Audio reliability",
            _percent(audio.get("reliability", 0)),
        ],
        [
            "Cross-modal relationship",
            _safe(relationship),
        ],
        [
            "Agreement score",
            _percent(fusion.get("agreement_score", 0)),
        ],
        [
            "Evidence strength",
            _percent(evidence_strength)
            if evidence_strength is not None
            else "N/A",
        ],
    ]

    table = Table(summary_data, colWidths=[2.3 * inch, 4.3 * inch])

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.whitesmoke, colors.lightgrey]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(table)

    # =========================================================
    # DECISION EXPLANATION
    # =========================================================

    story.append(
        Paragraph("2. Why AViGuard Reached This Decision", heading_style)
    )

    visual_conf = visual.get(
        "confidence",
        visual.get("top_confidence", 0),
    )

    audio_conf = audio.get(
        "confidence",
        audio.get("top_confidence", 0),
    )

    visual_rel = visual.get("reliability", 0)
    audio_rel = audio.get("reliability", 0)

    visual_weight = fusion.get("visual_weight", 0)
    audio_weight = fusion.get("audio_weight", 0)

    explanation = f"""
    AViGuard does not treat the video and audio streams as equally reliable by
    default. The system first analyzes the two modalities independently and
    then combines their evidence.
    """

    story.append(Paragraph(explanation, body_style))

    explanation_points = [
        f"<b>Visual evidence:</b> The visual model identified "
        f"<b>{_safe(visual_label)}</b> with confidence "
        f"<b>{_percent(visual_conf)}</b>.",

        f"<b>Visual reliability:</b> The estimated reliability of the "
        f"visual stream was <b>{_percent(visual_rel)}</b>, incorporating "
        f"model confidence and visual signal quality.",

        f"<b>Acoustic evidence:</b> The acoustic model identified "
        f"<b>{_safe(audio_label)}</b> with confidence "
        f"<b>{_percent(audio_conf)}</b>.",

        f"<b>Audio reliability:</b> The estimated reliability of the "
        f"audio stream was <b>{_percent(audio_rel)}</b>.",

        f"<b>Adaptive fusion:</b> The visual modality contributed "
        f"<b>{_percent(visual_weight)}</b> while the audio modality "
        f"contributed <b>{_percent(audio_weight)}</b>.",

        f"<b>Cross-modal consistency:</b> The two independent "
        f"interpretations were compared in a shared semantic space. "
        f"The resulting relationship was <b>{_safe(relationship)}</b>.",
    ]

    for point in explanation_points:
        story.append(
            Paragraph("• " + point, body_style)
        )

    # =========================================================
    # VISUAL ANALYSIS
    # =========================================================

    story.append(
        Paragraph("3. Visual Analysis", heading_style)
    )

    visual_rows = [
        ["Metric", "Value"],
        [
            "Top prediction",
            _safe(visual_label),
        ],
        [
            "Confidence",
            _percent(visual_conf),
        ],
        [
            "Reliability",
            _percent(visual_rel),
        ],
        [
            "Brightness",
            _safe(visual.get("brightness")),
        ],
        [
            "Brightness quality",
            _safe(visual.get("brightness_quality")),
        ],
        [
            "Sharpness",
            _safe(visual.get("sharpness")),
        ],
        [
            "Sharpness quality",
            _safe(visual.get("sharpness_quality")),
        ],
        [
            "Overall visual quality",
            _safe(visual.get("quality")),
        ],
    ]

    vt = Table(visual_rows, colWidths=[2.5 * inch, 4.1 * inch])

    vt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.whitesmoke, colors.lightgrey]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(vt)

    # =========================================================
    # VISUAL PREDICTIONS
    # =========================================================

    story.append(
        Paragraph("Top Visual Predictions", heading_style)
    )

    predictions = visual.get("predictions", [])

    prediction_rows = [["Rank", "Prediction", "Confidence"]]

    for i, item in enumerate(predictions[:5], start=1):

        if isinstance(item, dict):
            label = item.get(
                "label",
                item.get("name", "Unknown"),
            )

            confidence = item.get(
                "confidence",
                item.get("score", 0),
            )

        else:
            try:
                label, confidence = item
            except Exception:
                label = str(item)
                confidence = 0

        prediction_rows.append(
            [
                str(i),
                str(label),
                _percent(confidence),
            ]
        )

    if len(prediction_rows) > 1:
        pt = Table(
            prediction_rows,
            colWidths=[0.7 * inch, 3.8 * inch, 2.1 * inch],
        )

        pt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0),
                     colors.HexColor("#374151")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )

        story.append(pt)

    # =========================================================
    # AUDIO
    # =========================================================

    story.append(
        Paragraph("4. Acoustic Analysis", heading_style)
    )

    audio_rows = [
        ["Metric", "Value"],
        ["Audio available", _safe(audio.get("has_audio"))],
        ["Top prediction", _safe(audio_label)],
        ["Confidence", _percent(audio_conf)],
        ["Reliability", _percent(audio_rel)],
        ["Sample rate", _safe(audio.get("sample_rate"))],
        ["Duration", _safe(audio.get("duration"))],
        ["RMS energy", _safe(audio.get("rms"))],
        ["Spectral centroid", _safe(audio.get("spectral_centroid"))],
        ["Zero crossing rate", _safe(audio.get("zero_crossing_rate"))],
        ["Silence ratio", _percent(audio.get("silence_ratio", 0))],
        ["Non-silent ratio", _percent(audio.get("non_silent_ratio", 0))],
        ["Audio quality", _safe(audio.get("quality"))],
    ]

    at = Table(audio_rows, colWidths=[2.5 * inch, 4.1 * inch])

    at.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#374151")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.whitesmoke, colors.lightgrey]),
            ]
        )
    )

    story.append(at)

    # =========================================================
    # AUDIO PREDICTIONS
    # =========================================================

    story.append(
        Paragraph("Top Acoustic Predictions", heading_style)
    )

    audio_predictions = audio.get("predictions", [])

    ap_rows = [["Rank", "Prediction", "Confidence"]]

    for i, item in enumerate(audio_predictions[:5], start=1):

        if isinstance(item, dict):
            label = item.get(
                "label",
                item.get("name", "Unknown"),
            )

            confidence = item.get(
                "confidence",
                item.get("score", 0),
            )

        else:
            try:
                label, confidence = item
            except Exception:
                label = str(item)
                confidence = 0

        ap_rows.append(
            [
                str(i),
                str(label),
                _percent(confidence),
            ]
        )

    if len(ap_rows) > 1:

        apt = Table(
            ap_rows,
            colWidths=[0.7 * inch, 3.8 * inch, 2.1 * inch],
        )

        apt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0),
                     colors.HexColor("#374151")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )

        story.append(apt)

    # =========================================================
    # WAVEFORM
    # =========================================================

    waveform = audio.get("waveform")

    waveform_image = create_waveform_image(
        waveform,
        audio.get("sample_rate", 16000),
    )

    if waveform_image is not None:

        story.append(
            Paragraph("Acoustic Waveform", heading_style)
        )

        story.append(
            Image(
                waveform_image,
                width=7 * inch,
                height=2.2 * inch,
            )
        )

    # =========================================================
    # FUSION
    # =========================================================

    story.append(
        Paragraph("5. Reliability-Aware Fusion", heading_style)
    )

    fusion_rows = [
        ["Fusion metric", "Value"],
        [
            "Visual contribution",
            _percent(visual_weight),
        ],
        [
            "Audio contribution",
            _percent(audio_weight),
        ],
        [
            "Agreement score",
            _percent(fusion.get("agreement_score", 0)),
        ],
        [
            "Relationship",
            _safe(relationship),
        ],
        [
            "Evidence strength",
            _percent(evidence_strength)
            if evidence_strength is not None
            else "N/A",
        ],
        [
            "Context",
            _safe(fusion.get("context")),
        ],
    ]

    ft = Table(
        fusion_rows,
        colWidths=[2.5 * inch, 4.1 * inch],
    )

    ft.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#374151")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.whitesmoke, colors.lightgrey]),
            ]
        )
    )

    story.append(ft)

    # =========================================================
    # FRAMES
    # =========================================================

    if frame_images:

        story.append(PageBreak())

        story.append(
            Paragraph(
                "6. Sampled Visual Evidence",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                "The following frames were sampled from the analyzed "
                "video to provide visual evidence supporting the temporal "
                "video inference.",
                body_style,
            )
        )

        for frame in frame_images[:6]:

            try:
                story.append(
                    Image(
                        frame,
                        width=3.0 * inch,
                        height=2.0 * inch,
                    )
                )

                story.append(Spacer(1, 5))

            except Exception:
                pass

    # =========================================================
    # SYSTEM DESCRIPTION
    # =========================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "7. AViGuard System Architecture",
            heading_style,
        )
    )

    architecture = """
    <b>Video branch:</b> R3D-18 pretrained on Kinetics-400. The video
    pipeline samples temporal frames, performs spatial preprocessing and
    extracts visual-temporal evidence.

    <br/><br/>

    <b>Audio branch:</b> Audio Spectrogram Transformer (AST) pretrained
    on AudioSet. The audio pipeline extracts a 16 kHz mono waveform using
    FFmpeg and processes acoustic information for classification.

    <br/><br/>

    <b>Quality analysis:</b> AViGuard estimates visual quality using
    brightness and sharpness measurements and acoustic quality using
    signal-level statistics.

    <br/><br/>

    <b>Fusion:</b> The independent modality outputs are combined using
    reliability-aware adaptive weighting rather than assuming that both
    modalities are equally trustworthy.

    <br/><br/>

    <b>Consistency analysis:</b> The predicted concepts are mapped into
    broad semantic categories and compared to determine whether the two
    evidence streams support or disagree with each other.
    """

    story.append(Paragraph(architecture, body_style))

    story.append(
        Paragraph("Important interpretation note", heading_style)
    )

    story.append(
        Paragraph(
            "The evidence strength and reliability values are engineering "
            "scores used by AViGuard for evidence aggregation. They should "
            "not be interpreted as calibrated probabilities.",
            body_style,
        )
    )

    # =========================================================
    # BUILD
    # =========================================================

    doc.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()