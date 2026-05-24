from typing import Any

from pydantic import BaseModel


class VerifyResponse(BaseModel):
  success: bool
  liveness_passed: bool
  face_match_passed: bool
  single_face_detected: bool
  challenge_passed: bool
  liveness_score: float
  face_match_score: float
  face_match_metric: str = "cosine"
  liveness_threshold: float
  face_match_threshold: float
  failure_reason: str = ""
  quality_checks: dict[str, Any] | None = None
  provider: str = "opencv-yunet-sface-minifasnet"
