import cv2
import numpy as np

from .config import settings
from .face_detector import ModelMissingError


class FaceVerifier:
  def __init__(self) -> None:
    if not settings.sface_model_path.exists():
      raise ModelMissingError(
        f"Missing SFace face recognition model: {settings.sface_model_path}"
      )

    self.recognizer = cv2.FaceRecognizerSF_create(str(settings.sface_model_path), "")

  def compare(
    self,
    passport_image: np.ndarray,
    passport_face: dict,
    live_image: np.ndarray,
    live_face: dict,
  ) -> tuple[float, str]:
    passport_aligned = self.recognizer.alignCrop(
      passport_image,
      passport_face["raw"].astype(np.float32),
    )
    live_aligned = self.recognizer.alignCrop(
      live_image,
      live_face["raw"].astype(np.float32),
    )

    passport_feature = self.recognizer.feature(passport_aligned)
    live_feature = self.recognizer.feature(live_aligned)

    metric = settings.face_match_metric
    opencv_metric = (
      cv2.FaceRecognizerSF_FR_NORM_L2
      if metric == "l2"
      else cv2.FaceRecognizerSF_FR_COSINE
    )

    score = self.recognizer.match(
      passport_feature,
      live_feature,
      opencv_metric,
    )

    return float(score), metric
