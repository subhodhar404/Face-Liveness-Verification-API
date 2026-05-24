import cv2
import numpy as np

from .config import settings


class ModelMissingError(RuntimeError):
  pass


class FaceDetector:
  def __init__(self) -> None:
    if not settings.yunet_model_path.exists():
      raise ModelMissingError(
        f"Missing YuNet face detection model: {settings.yunet_model_path}"
      )

    self.detector = cv2.FaceDetectorYN_create(
      str(settings.yunet_model_path),
      "",
      (320, 320),
      0.9,
      0.3,
      5000,
    )

  def detect(self, image: np.ndarray) -> list[dict]:
    height, width = image.shape[:2]
    self.detector.setInputSize((width, height))
    _, faces = self.detector.detect(image)

    if faces is None:
      return []

    detected_faces = []

    for face in faces:
      raw = face.astype(np.float32)
      detected_faces.append(
        {
          "box": raw[0:4].tolist(),
          "landmarks": raw[4:14].reshape(5, 2).tolist(),
          "score": float(raw[14]),
          "raw": raw,
        }
      )

    return detected_faces
