# AViGuard

## Audio-Visual Event Detection & Adaptive Fusion

AViGuard is an end-to-end **audio-visual machine learning application** that analyzes video and audio independently, estimates the reliability of each modality, and adaptively combines their evidence to produce an interpretable analysis.

**Live Demo:** https://aviguard-multimodal.streamlit.app/

---

## Overview

Audio and video provide complementary information about the same real-world event. AViGuard treats them as independent evidence sources and evaluates how reliable each modality is before combining their outputs.

For an uploaded video, AViGuard:

- Extracts and analyzes visual information from sampled video frames.
- Extracts and analyzes the audio stream when available.
- Runs pretrained visual and audio classification models.
- Measures video and audio signal quality.
- Estimates modality reliability.
- Performs reliability-aware adaptive evidence fusion.
- Evaluates cross-modal semantic consistency.
- Presents the analysis through an interactive Streamlit application.

---

## System Architecture

```text
                    ┌──────────────────────┐
                    │   Video / Audio File  │
                    │      MP4 / WEBM       │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
       ┌──────────────────┐          ┌──────────────────┐
       │  Video Pipeline  │          │  Audio Pipeline  │
       │                  │          │                  │
       │ OpenCV           │          │ FFmpeg           │
       │ Frame Sampling   │          │ Librosa          │
       │ Quality Metrics  │          │ Audio Features   │
       └────────┬─────────┘          └────────┬─────────┘
                │                             │
                ▼                             ▼
       ┌──────────────────┐          ┌──────────────────┐
       │     R3D-18       │          │       AST        │
       │   Kinetics-400   │          │     AudioSet     │
       │ Visual Inference │          │ Audio Inference  │
       └────────┬─────────┘          └────────┬─────────┘
                │                             │
                ▼                             ▼
       ┌──────────────────┐          ┌──────────────────┐
       │ Visual Evidence  │          │ Audio Evidence   │
       │ Confidence       │          │ Confidence       │
       │ + Quality        │          │ + Quality        │
       └────────┬─────────┘          └────────┬─────────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
                 ┌──────────────────────────┐
                 │ Modality Reliability     │
                 │       Estimation         │
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ Reliability-Aware        │
                 │ Adaptive Fusion          │
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ Cross-Modal Semantic     │
                 │ Consistency Analysis     │
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ Interpretable Analysis   │
                 │ + Diagnostics + Report    │
                 └──────────────────────────┘

'''
Key Features
Video Analysis
Video frame sampling using OpenCV
Pretrained R3D-18 video classification
Kinetics-400 action recognition
Visual confidence estimation
Video brightness analysis
Video sharpness analysis
Audio Analysis
Audio extraction using FFmpeg
16 kHz mono waveform preprocessing
Audio signal analysis using Librosa
Pretrained Audio Spectrogram Transformer (AST)
AudioSet-based acoustic-event classification
RMS energy analysis
Non-silent ratio analysis
Graceful handling of videos without an audio stream
Reliability-Aware Fusion

AViGuard does not simply assign equal weights to audio and video.

The system estimates modality reliability using model confidence and signal-quality information.

Model Confidence
       +
Signal Quality
       │
       ▼
Modality Reliability
       │
       ▼
Fusion Weight

The resulting weights determine how strongly visual and acoustic evidence contributes to the combined analysis.

Cross-Modal Consistency

Predictions from the visual and audio models are mapped into shared semantic categories to evaluate whether the two modalities provide consistent evidence.

Example:

Visual Evidence
Playing Guitar
      │
      ▼
  Music Category
      ▲
      │
Audio Evidence
Music

This allows the system to report agreement or disagreement between the modalities.

Interpretability

The application provides:

Top visual predictions
Top audio predictions
Model confidence
Modality reliability
Fusion contribution
Cross-modal agreement
Video quality diagnostics
Audio quality diagnostics
Acoustic waveform visualization
Sampled video-frame inspection
Decision explanation
PDF analysis reports

Models
Modality	Model	Pretraining
Video	R3D-18	Kinetics-400
Audio	Audio Spectrogram Transformer (AST)	AudioSet

The pretrained models provide modality-specific predictions. AViGuard adds the reliability estimation, adaptive fusion, and cross-modal consistency analysis layer on top of these outputs.

Technology Stack
Machine Learning
Python
PyTorch
Torchvision
Hugging Face Transformers
Audio Processing
Librosa
SoundFile
FFmpeg
NumPy
SciPy
Video Processing
OpenCV
Torchvision video models
Video frame sampling
Image preprocessing
Application
Streamlit
Matplotlib
ReportLab
Development
Git
GitHub
Project Structure
AViGuard/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   │
│   ├── audio/
│   │   ├── __init__.py
│   │   └── audio_analyzer.py
│   │
│   ├── video/
│   │   ├── __init__.py
│   │   └── video_analyzer.py
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── fusion_engine.py
│   │
│   ├── models/
│   │   └── __init__.py
│   │
│   └── utils/
│       └── __init__.py
│
├── assets/
├── outputs/
└── tests/
Installation
1. Clone the repository
git clone https://github.com/Rushikannan2/AViGuard.git
cd AViGuard
2. Create a virtual environment
python -m venv .venv
3. Activate the environment
Windows
.venv\Scripts\activate
Linux / macOS
source .venv/bin/activate
4. Install dependencies
pip install -r requirements.txt
Run Locally

Start the Streamlit application:

streamlit run app.py

The application will be available at the local Streamlit URL displayed in the terminal.

Live Demo

Try the deployed application:

https://aviguard-multimodal.streamlit.app/

Example Workflow
Input Video
     │
     ├──► Video Frames ──► OpenCV ──► R3D-18 ──► Visual Evidence
     │
     └──► Audio ──► FFmpeg ──► Librosa / AST ──► Audio Evidence
                                               │
                                               ▼
                                      Quality Assessment
                                               │
                                               ▼
                                      Reliability Estimation
                                               │
                                               ▼
                                         Adaptive Fusion
                                               │
                                               ▼
                                    Cross-Modal Consistency
                                               │
                                               ▼
                                      Interpretable Report
Example Output

For a video containing a musical performance, AViGuard can independently identify visual and acoustic evidence such as:

Visual Prediction
Playing Guitar

Audio Prediction
Music

Cross-Modal Agreement
High

Fusion
Visual + Audio Reliability-Weighted Evidence

The exact predictions, confidence values, reliability values, and fusion contributions depend on the uploaded media.

Reliability-Aware Fusion

A fixed 50/50 fusion strategy can be unreliable when one modality is degraded.

AViGuard therefore considers both model confidence and signal quality when estimating the contribution of each modality.

Conceptually:

Visual Confidence ───────┐
                         ├──► Visual Reliability
Video Quality ───────────┘

Audio Confidence ────────┐
                         ├──► Audio Reliability
Audio Quality ───────────┘

Visual Reliability ──────┐
                         ├──► Adaptive Fusion
Audio Reliability ───────┘

This allows a weaker modality to contribute less evidence when its estimated reliability is low.

Missing Audio Handling

AViGuard supports videos that do not contain a usable audio stream.

Video
  │
  ├──► Visual Analysis ──► Available
  │
  └──► Audio Stream ─────► Not Available
                              │
                              ▼
                       Audio Reliability = 0

The visual analysis can continue without requiring an audio stream.

Limitations
The visual and audio classifiers are pretrained models and are not trained specifically on a custom AViGuard dataset.
Cross-modal consistency uses a semantic category mapping rather than a learned multimodal neural network.
Fusion scores represent evidence weighting and should not be interpreted as calibrated probabilities.
Classification performance depends on the capabilities and label spaces of the underlying pretrained models.
CPU inference can be slower than GPU inference for larger videos.
Future Improvements
Fine-tune modality-specific models on domain-specific audio-visual datasets.
Learn the fusion function using supervised multimodal training data.
Add temporal event localization.
Explore joint audio-visual representation learning.
Add systematic benchmark evaluation using multimodal classification metrics.
Optimize model loading and inference for production deployment.