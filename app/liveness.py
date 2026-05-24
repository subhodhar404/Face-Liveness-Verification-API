import json
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .config import settings
from .face_detector import FaceDetector, ModelMissingError
from .utils import clamp_crop, softmax


class LivenessDetector:
  def __init__(self) -> None:
    if not settings.liveness_model_path.exists():
      raise ModelMissingError(
        f"Missing liveness ONNX model: {settings.liveness_model_path}"
      )

    self.session = ort.InferenceSession(
      str(settings.liveness_model_path),
      providers=["CPUExecutionProvider"],
    )
    self.input_name = self.session.get_inputs()[0].name
    self.input_shape = self.session.get_inputs()[0].shape

  def _input_size(self) -> tuple[int, int]:
    shape = self.input_shape

    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
      return shape[3], shape[2]

    if len(shape) == 4 and isinstance(shape[1], int) and isinstance(shape[2], int):
      return shape[2], shape[1]

    return 80, 80

  def score(self, image: np.ndarray, face: dict) -> float:
    crop = clamp_crop(image, face["box"])
    width, height = self._input_size()
    resized = cv2.resize(crop, (width, height))
    normalized = resized.astype(np.float32) / 255.0
    nchw = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

    outputs = self.session.run(None, {self.input_name: nchw})
    raw_output = np.asarray(outputs[0]).flatten()

    if raw_output.size >= 2:
      probabilities = softmax(raw_output[:2])
      return float(probabilities[-1])

    if raw_output.size == 1:
      return float(1.0 / (1.0 + np.exp(-raw_output[0])))

    return 0.0


class ChallengeVerifier:
  def __init__(self, detector: FaceDetector) -> None:
    self.detector = detector

  def verify(self, challenge_frames: list[np.ndarray], metadata_raw: str) -> tuple[bool, str]:
    try:
      metadata: dict[str, Any] = json.loads(metadata_raw or "{}")
    except json.JSONDecodeError:
      metadata = {}

    expected_challenges = metadata.get("challenges") or [
      "blink",
      "turn_left",
      "turn_right",
      "smile",
    ]

    if len(challenge_frames) < len(expected_challenges):
      return False, "Not enough challenge frames were provided"

    single_face_frame_count = 0

    for frame in challenge_frames:
      faces = self.detector.detect(frame)

      if len(faces) == 1:
        single_face_frame_count += 1

    required_count = max(len(expected_challenges), len(challenge_frames) // 2)

    if single_face_frame_count < required_count:
      return False, "Challenge frames did not consistently contain one face"

    return True, ""
