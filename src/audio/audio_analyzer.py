import os
import shutil
import subprocess
import tempfile

import imageio_ffmpeg
import librosa
import numpy as np
import torch

from transformers import (
    AutoProcessor,
    AutoModelForAudioClassification,
)


class AudioAnalyzer:
    """
    AViGuard Audio Intelligence.

    Supports:
        1. Videos containing audio
        2. Video-only files

    Audio pipeline:

        Video
          ↓
        FFmpeg
          ↓
        16 kHz mono WAV
          ↓
        Librosa
          ↓
        AST / AudioSet
          ↓
        Audio prediction
          ↓
        Audio quality
          ↓
        Audio reliability

    If the input contains no audio stream, the analyzer
    returns a graceful 'audio unavailable' result instead
    of crashing the complete multimodal pipeline.
    """

    MODEL_NAME = (
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    SAMPLE_RATE = 16000
    MAX_DURATION = 10

    def __init__(self, device=None):

        # -------------------------------------------------
        # DEVICE
        # -------------------------------------------------

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[AudioAnalyzer] Device: {self.device}"
        )

        # -------------------------------------------------
        # AST
        # -------------------------------------------------

        print(
            "[AudioAnalyzer] Loading AST..."
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                self.MODEL_NAME
            )
        )

        self.model = (
            AutoModelForAudioClassification.from_pretrained(
                self.MODEL_NAME
            )
        )

        self.model.eval()
        self.model.to(self.device)

        self.labels = self.model.config.id2label

        print(
            "[AudioAnalyzer] AST loaded."
        )

    # =====================================================
    # CHECK WHETHER VIDEO CONTAINS AUDIO
    # =====================================================

    def _has_audio_stream(self, video_path):
        """
        Determine whether the input contains an audio stream.

        Returns:
            True  -> audio stream exists
            False -> video-only / no usable audio stream
        """

        ffmpeg_exe = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

        command = [
            ffmpeg_exe,
            "-hide_banner",
            "-i",
            video_path,
        ]

        try:

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            output = (
                result.stdout
                + "\n"
                + result.stderr
            ).lower()

            return (
                "audio:" in output
                or " audio " in output
                or "audio stream" in output
            )

        except Exception:

            return False

    # =====================================================
    # EXTRACT AUDIO
    # =====================================================

    def _extract_audio(self, video_path):

        if not os.path.isfile(video_path):

            raise FileNotFoundError(
                f"Video file not found: {video_path}"
            )

        # -------------------------------------------------
        # First check for an audio stream.
        # -------------------------------------------------

        print(
            "[AudioAnalyzer] Checking for audio stream..."
        )

        if not self._has_audio_stream(video_path):

            print(
                "[AudioAnalyzer] No audio stream detected."
            )

            return None, None

        # -------------------------------------------------
        # FFmpeg
        # -------------------------------------------------

        ffmpeg_exe = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

        temp_dir = tempfile.mkdtemp(
            prefix="aviguard_audio_"
        )

        audio_path = os.path.join(
            temp_dir,
            "audio.wav"
        )

        command = [
            ffmpeg_exe,

            "-y",

            "-hide_banner",

            "-loglevel",
            "error",

            "-i",
            video_path,

            # Disable video
            "-vn",

            # Mono
            "-ac",
            "1",

            # 16 kHz
            "-ar",
            str(self.SAMPLE_RATE),

            # PCM WAV
            "-c:a",
            "pcm_s16le",

            audio_path,
        ]

        print(
            "[AudioAnalyzer] Extracting audio..."
        )

        try:

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

        except Exception as exc:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )

            raise RuntimeError(
                f"Could not execute FFmpeg: {exc}"
            ) from exc

        # -------------------------------------------------
        # FFmpeg failure
        # -------------------------------------------------

        if result.returncode != 0:

            error_text = (
                result.stderr.strip()
            )

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )

            # Some unusual containers may report that no
            # audio stream is available.
            if (
                "does not contain any stream"
                in error_text.lower()
                or
                "matches no streams"
                in error_text.lower()
                or
                "no streams" in error_text.lower()
            ):

                print(
                    "[AudioAnalyzer] No usable audio stream."
                )

                return None, None

            raise RuntimeError(
                "FFmpeg audio extraction failed:\n\n"
                + error_text[-3000:]
            )

        # -------------------------------------------------
        # Validate WAV
        # -------------------------------------------------

        if not os.path.isfile(audio_path):

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )

            raise RuntimeError(
                "FFmpeg finished but no WAV file was created."
            )

        if os.path.getsize(audio_path) == 0:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )

            raise RuntimeError(
                "FFmpeg created an empty WAV file."
            )

        print(
            "[AudioAnalyzer] Audio extraction successful."
        )

        return audio_path, temp_dir

    # =====================================================
    # CATEGORY
    # =====================================================

    @staticmethod
    def _category(label):

        text = str(label).lower()

        groups = {

            "speech": [
                "speech",
                "conversation",
                "narration",
                "talking",
                "speaking",
                "voice",
                "dialogue",
                "whisper",
                "monologue",
            ],

            "music": [
                "music",
                "guitar",
                "piano",
                "violin",
                "drum",
                "singing",
                "song",
                "concert",
                "instrument",
                "orchestra",
                "melody",
                "jazz",
                "rock music",
                "pop music",
            ],

            "vehicle": [
                "vehicle",
                "car",
                "engine",
                "motor",
                "truck",
                "bus",
                "train",
                "aircraft",
                "airplane",
                "motorcycle",
                "traffic",
                "automobile",
                "horn",
                "road",
            ],

            "animal": [
                "dog",
                "cat",
                "bird",
                "horse",
                "animal",
                "bark",
                "meow",
                "roar",
                "growl",
                "insect",
                "chirp",
                "crow",
            ],

            "sports": [
                "sport",
                "stadium",
                "ball",
                "cheering",
                "applause",
                "crowd",
                "football",
                "basketball",
                "soccer",
                "tennis",
                "baseball",
            ],

            "nature": [
                "rain",
                "water",
                "wind",
                "thunder",
                "fire",
                "nature",
                "ocean",
                "wave",
                "river",
                "stream",
            ],

            "food": [
                "cooking",
                "frying",
                "food",
                "kitchen",
                "restaurant",
                "cutlery",
            ],
        }

        for category, keywords in groups.items():

            if any(
                keyword in text
                for keyword in keywords
            ):

                return category

        return "other"

    # =====================================================
    # VIDEO-ONLY RESULT
    # =====================================================

    @staticmethod
    def _no_audio_result():

        return {

            "has_audio": False,

            "sample_rate": 0,

            "duration": 0.0,

            "rms": 0.0,

            "spectral_centroid": 0.0,

            "zero_crossing_rate": 0.0,

            "silence_ratio": 1.0,

            "non_silent_ratio": 0.0,

            "quality": 0.0,

            "reliability": 0.0,

            "top_prediction": {
                "label": "No audio stream",
                "confidence": 0.0,
                "category": "other",
            },

            "predictions": [],

            "waveform": np.array(
                [],
                dtype=np.float32
            ),

            "device": self.device,

            "status":
                "The uploaded video does not contain "
                "a usable audio stream.",
        }

    # =====================================================
    # QUALITY
    # =====================================================

    @staticmethod
    def _calculate_quality(
        rms_mean,
        silence_ratio,
    ):

        rms_quality = min(
            1.0,
            float(rms_mean) / 0.10
        )

        non_silent_ratio = (
            1.0 - float(silence_ratio)
        )

        quality = (
            0.6 * rms_quality
            + 0.4 * non_silent_ratio
        )

        return float(
            np.clip(
                quality,
                0.0,
                1.0
            )
        )

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def analyze(self, video_path):

        audio_path = None
        temp_dir = None

        # -------------------------------------------------
        # Extract audio
        # -------------------------------------------------

        (
            audio_path,
            temp_dir,
        ) = self._extract_audio(
            video_path
        )

        # -------------------------------------------------
        # Video-only input
        # -------------------------------------------------

        if audio_path is None:

            return self._no_audio_result()

        # -------------------------------------------------
        # Load waveform
        # -------------------------------------------------

        try:

            print(
                "[AudioAnalyzer] Loading waveform..."
            )

            waveform, sample_rate = (
                librosa.load(
                    audio_path,
                    sr=self.SAMPLE_RATE,
                    mono=True,
                )
            )

        finally:

            if temp_dir is not None:

                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True
                )

        # -------------------------------------------------
        # Validate
        # -------------------------------------------------

        if (
            waveform is None
            or len(waveform) == 0
        ):

            return self._no_audio_result()

        # -------------------------------------------------
        # Limit to 10 seconds
        # -------------------------------------------------

        max_samples = (
            self.SAMPLE_RATE
            * self.MAX_DURATION
        )

        if len(waveform) > max_samples:

            waveform = waveform[
                :max_samples
            ]

        duration = (
            len(waveform)
            / sample_rate
        )

        # =================================================
        # SIGNAL FEATURES
        # =================================================

        print(
            "[AudioAnalyzer] Computing audio features..."
        )

        rms = librosa.feature.rms(
            y=waveform
        )[0]

        spectral_centroid = (
            librosa.feature.spectral_centroid(
                y=waveform,
                sr=sample_rate,
            )[0]
        )

        zero_crossing_rate = (
            librosa.feature.zero_crossing_rate(
                waveform
            )[0]
        )

        silence_threshold = 0.01

        silence_ratio = float(
            np.mean(
                np.abs(waveform)
                < silence_threshold
            )
        )

        non_silent_ratio = (
            1.0 - silence_ratio
        )

        rms_mean = float(
            np.mean(rms)
        )

        centroid_mean = float(
            np.mean(
                spectral_centroid
            )
        )

        zcr_mean = float(
            np.mean(
                zero_crossing_rate
            )
        )

        # =================================================
        # AST
        # =================================================

        print(
            "[AudioAnalyzer] Running AST..."
        )

        inputs = self.processor(
            waveform,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        # =================================================
        # TOP 5
        # =================================================

        k = min(
            5,
            probabilities.shape[-1]
        )

        values, indices = torch.topk(
            probabilities,
            k=k,
        )

        values_np = (
            values.detach()
            .cpu()
            .numpy()
        )

        indices_np = (
            indices.detach()
            .cpu()
            .numpy()
        )

        predictions = []

        for value, index in zip(
            values_np,
            indices_np,
        ):

            index = int(index)

            if isinstance(
                self.labels,
                dict
            ):

                label = self.labels.get(
                    index,
                    str(index)
                )

            else:

                label = str(index)

            predictions.append(
                {
                    "label": str(label),

                    "confidence": float(
                        value
                    ),

                    "category": (
                        self._category(
                            label
                        )
                    ),
                }
            )

        if not predictions:

            raise RuntimeError(
                "AST did not produce predictions."
            )

        top_prediction = predictions[0]

        # =================================================
        # QUALITY
        # =================================================

        audio_quality = (
            self._calculate_quality(
                rms_mean,
                silence_ratio,
            )
        )

        # =================================================
        # RELIABILITY
        # =================================================

        audio_reliability = (
            0.6
            * top_prediction["confidence"]
            + 0.4
            * audio_quality
        )

        audio_reliability = float(
            np.clip(
                audio_reliability,
                0.0,
                1.0
            )
        )

        # =================================================
        # RESULT
        # =================================================

        return {

            "has_audio": True,

            "sample_rate":
                int(sample_rate),

            "duration":
                float(duration),

            "rms":
                rms_mean,

            "spectral_centroid":
                centroid_mean,

            "zero_crossing_rate":
                zcr_mean,

            "silence_ratio":
                silence_ratio,

            "non_silent_ratio":
                float(non_silent_ratio),

            "quality":
                float(audio_quality),

            "reliability":
                float(audio_reliability),

            "top_prediction":
                top_prediction,

            "predictions":
                predictions,

            "waveform":
                waveform,

            "device":
                self.device,

            "status":
                "Audio successfully analyzed.",
        }