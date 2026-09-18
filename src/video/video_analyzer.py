import cv2
import numpy as np
import torch

from PIL import Image
from torchvision.models.video import r3d_18, R3D_18_Weights


class VideoAnalyzer:
    """
    Temporal video intelligence using pretrained R3D-18.

    Outputs:
        - video metadata
        - top Kinetics action predictions
        - brightness quality
        - sharpness quality
        - overall video quality
        - visual reliability
    """

    def __init__(self, device=None):

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[VideoAnalyzer] Device: {self.device}"
        )

        print(
            "[VideoAnalyzer] Loading R3D-18..."
        )

        self.weights = (
            R3D_18_Weights.KINETICS400_V1
        )

        self.model = r3d_18(
            weights=self.weights
        )

        self.model.eval()
        self.model.to(self.device)

        self.labels = (
            self.weights.meta["categories"]
        )

        # Kinetics normalization values.
        self.mean = torch.tensor(
            [
                0.43216,
                0.394666,
                0.37645,
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1, 1)

        self.std = torch.tensor(
            [
                0.22803,
                0.22145,
                0.216989,
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1, 1)

    @staticmethod
    def _category(label):

        text = label.lower()

        category_keywords = {

            "music": [
                "guitar",
                "violin",
                "piano",
                "drum",
                "music",
                "sing",
                "concert",
                "instrument",
            ],

            "speech": [
                "talk",
                "talking",
                "speaking",
                "conversation",
                "phone",
                "speech",
            ],

            "vehicle": [
                "car",
                "driving",
                "motorcycle",
                "bicycle",
                "bike",
                "bus",
                "truck",
                "train",
                "airplane",
                "helicopter",
                "vehicle",
            ],

            "sports": [
                "basketball",
                "football",
                "soccer",
                "tennis",
                "volleyball",
                "baseball",
                "golf",
                "swimming",
                "skiing",
                "skateboarding",
                "sport",
            ],

            "animal": [
                "dog",
                "cat",
                "horse",
                "bird",
                "animal",
                "cow",
                "fish",
            ],

            "food": [
                "cooking",
                "eating",
                "baking",
                "food",
                "cake",
                "pizza",
            ],

            "person": [
                "walking",
                "running",
                "jumping",
                "clapping",
                "handshake",
                "shaking hands",
                "hugging",
                "people",
                "person",
            ],

            "nature": [
                "water",
                "river",
                "ocean",
                "rain",
                "snow",
                "fire",
                "forest",
                "nature",
            ],
        }

        for category, keywords in (
            category_keywords.items()
        ):

            if any(
                keyword in text
                for keyword in keywords
            ):
                return category

        return "other"

    @staticmethod
    def _prepare_frame(frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        image = image.resize(
            (128, 128)
        )

        # Center crop 128 -> 112.
        left = 8
        top = 8

        image = image.crop(
            (
                left,
                top,
                left + 112,
                top + 112,
            )
        )

        array = (
            np.asarray(image)
            .astype(np.float32)
            / 255.0
        )

        tensor = torch.from_numpy(
            array
        )

        # H,W,C -> C,H,W
        tensor = tensor.permute(
            2,
            0,
            1
        )

        return tensor

    def _prepare_video(self, frames):

        tensors = [
            self._prepare_frame(frame)
            for frame in frames
        ]

        # T,C,H,W
        video = torch.stack(
            tensors
        )

        # C,T,H,W
        video = video.permute(
            1,
            0,
            2,
            3
        )

        # B,C,T,H,W
        video = video.unsqueeze(0)

        video = video.to(
            self.device
        )

        video = (
            video - self.mean
        ) / self.std

        return video

    def analyze(
        self,
        video_path,
        num_frames=16
    ):

        capture = cv2.VideoCapture(
            video_path
        )

        if not capture.isOpened():
            raise ValueError(
                "Unable to open video."
            )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if fps <= 0:
            fps = 30.0

        if frame_count <= 0:
            capture.release()

            raise ValueError(
                "Video contains no readable frames."
            )

        duration = (
            frame_count / fps
        )

        sample_count = min(
            num_frames,
            frame_count
        )

        indices = np.linspace(
            0,
            frame_count - 1,
            sample_count,
            dtype=int
        )

        frames = []

        brightness_values = []
        sharpness_values = []

        for index in indices:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index)
            )

            success, frame = (
                capture.read()
            )

            if not success:
                continue

            frames.append(frame)

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            brightness_values.append(
                float(
                    gray.mean() / 255.0
                )
            )

            sharpness_values.append(
                float(
                    cv2.Laplacian(
                        gray,
                        cv2.CV_64F
                    ).var()
                )
            )

        capture.release()

        if not frames:
            raise ValueError(
                "Could not sample video frames."
            )

        # R3D expects a fixed temporal window.
        while len(frames) < 16:

            frames.append(
                frames[-1].copy()
            )

        frames = frames[:16]

        video_tensor = (
            self._prepare_video(frames)
        )

        with torch.inference_mode():

            logits = self.model(
                video_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

        k = min(
            5,
            len(self.labels)
        )

        values, indices = torch.topk(
            probabilities,
            k=k
        )

        predictions = []

        for value, index in zip(
            values.cpu().numpy(),
            indices.cpu().numpy()
        ):

            label = self.labels[
                int(index)
            ]

            predictions.append(
                {
                    "label": label,
                    "confidence": float(value),
                    "category": self._category(
                        label
                    ),
                }
            )

        top_prediction = (
            predictions[0]
        )

        mean_brightness = float(
            np.mean(
                brightness_values
            )
        )

        mean_sharpness = float(
            np.mean(
                sharpness_values
            )
        )

        # ---------------------------------
        # Quality heuristics
        # ---------------------------------

        brightness_quality = (
            1.0
            - min(
                1.0,
                abs(
                    mean_brightness - 0.5
                ) / 0.5
            )
        )

        sharpness_quality = min(
            1.0,
            np.log1p(
                mean_sharpness
            )
            / np.log1p(500.0)
        )

        video_quality = (
            0.5
            * brightness_quality
            +
            0.5
            * sharpness_quality
        )

        visual_reliability = (
            0.6
            * top_prediction[
                "confidence"
            ]
            +
            0.4
            * video_quality
        )

        return {

            "duration": float(
                duration
            ),

            "fps": float(
                fps
            ),

            "frame_count": int(
                frame_count
            ),

            "frames_analyzed": int(
                len(frames)
            ),

            "top_prediction":
                top_prediction,

            "predictions":
                predictions,

            "brightness":
                mean_brightness,

            "sharpness":
                mean_sharpness,

            "brightness_quality":
                float(
                    brightness_quality
                ),

            "sharpness_quality":
                float(
                    sharpness_quality
                ),

            "quality":
                float(
                    video_quality
                ),

            "reliability":
                float(
                    visual_reliability
                ),
        }