class FusionEngine:
    """
    Reliability-aware multimodal evidence fusion.

    Handles:
        - Video + audio
        - Video-only inputs
        - Audio-only / unavailable modality
    """

    RELATED_CATEGORIES = {
        ("person", "speech"),
        ("speech", "person"),
        ("person", "music"),
        ("music", "person"),
    }

    def fuse(self, visual, audio):

        visual_available = True

        audio_available = bool(
            audio.get(
                "has_audio",
                True
            )
        )

        # =================================================
        # VISUAL
        # =================================================

        visual_reliability = float(
            visual.get(
                "reliability",
                0.0
            )
        )

        visual_prediction = (
            visual.get(
                "top_prediction",
                {}
            )
        )

        visual_category = (
            visual_prediction.get(
                "category",
                "other"
            )
        )

        visual_label = (
            visual_prediction.get(
                "label",
                "Unknown"
            )
        )

        visual_confidence = float(
            visual_prediction.get(
                "confidence",
                0.0
            )
        )

        # =================================================
        # AUDIO
        # =================================================

        if audio_available:

            audio_reliability = float(
                audio.get(
                    "reliability",
                    0.0
                )
            )

            audio_prediction = (
                audio.get(
                    "top_prediction",
                    {}
                )
            )

            audio_category = (
                audio_prediction.get(
                    "category",
                    "other"
                )
            )

            audio_label = (
                audio_prediction.get(
                    "label",
                    "Unknown"
                )
            )

            audio_confidence = float(
                audio_prediction.get(
                    "confidence",
                    0.0
                )
            )

        else:

            audio_reliability = 0.0

            audio_category = "unavailable"

            audio_label = "No audio stream"

            audio_confidence = 0.0

        # =================================================
        # ADAPTIVE WEIGHTS
        # =================================================

        total_reliability = (
            visual_reliability
            + audio_reliability
        )

        if total_reliability > 1e-8:

            visual_weight = (
                visual_reliability
                / total_reliability
            )

            audio_weight = (
                audio_reliability
                / total_reliability
            )

        else:

            visual_weight = 1.0
            audio_weight = 0.0

        # =================================================
        # CROSS-MODAL CONSISTENCY
        # =================================================

        if not audio_available:

            agreement_score = 0.0

            agreement = (
                "Audio unavailable"
            )

            context = (
                "Only visual evidence is available. "
                "Cross-modal consistency cannot be "
                "estimated for this input."
            )

        elif (
            visual_category == audio_category
            and visual_category != "other"
        ):

            agreement_score = 0.90

            agreement = (
                "Strong agreement"
            )

            context = (
                f"Both modalities support a "
                f"{visual_category} context."
            )

        elif (
            visual_category,
            audio_category
        ) in self.RELATED_CATEGORIES:

            agreement_score = 0.65

            agreement = (
                "Related evidence"
            )

            context = (
                "The modalities provide related "
                "but not identical evidence."
            )

        elif (
            visual_category == "other"
            or audio_category == "other"
        ):

            agreement_score = 0.50

            agreement = "Uncertain"

            context = (
                "One or both modalities produced "
                "a broad or ambiguous category."
            )

        else:

            agreement_score = 0.15

            agreement = (
                "Potential disagreement"
            )

            context = (
                "The modalities provide different "
                "signals. This may indicate an "
                "off-screen sound, background noise, "
                "or visually ambiguous content."
            )

        # =================================================
        # EVIDENCE STRENGTH
        # =================================================

        evidence_strength = (
            visual_weight
            * visual_confidence
            +
            audio_weight
            * audio_confidence
        )

        if audio_available:

            consistency_adjusted_strength = (
                evidence_strength
                * (
                    0.55
                    + 0.45 * agreement_score
                )
            )

        else:

            # Do not penalize visual-only inference
            # using a nonexistent audio modality.
            consistency_adjusted_strength = (
                evidence_strength
            )

        # =================================================
        # RESULT
        # =================================================

        return {

            "visual_available":
                visual_available,

            "audio_available":
                audio_available,

            "visual_weight":
                float(visual_weight),

            "audio_weight":
                float(audio_weight),

            "visual_category":
                visual_category,

            "audio_category":
                audio_category,

            "visual_label":
                visual_label,

            "audio_label":
                audio_label,

            "agreement_score":
                float(agreement_score),

            "agreement":
                agreement,

            "evidence_strength":
                float(evidence_strength),

            "consistency_adjusted_strength":
                float(
                    consistency_adjusted_strength
                ),

            "context":
                context,
        }