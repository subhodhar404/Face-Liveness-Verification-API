from typing import List
import logging

from fastapi import Depends, FastAPI, File, Form, UploadFile

from .check_models import get_model_status
from .config import settings
from .face_detector import FaceDetector, ModelMissingError
from .face_verifier import FaceVerifier
from .liveness import ChallengeVerifier, LivenessDetector
from .quality import validate_image_quality
from .schemas import VerifyResponse
from .security import validate_ai_service_secret
from .utils import read_upload_image


app = FastAPI(title="FaceGuard API", version="1.0.0")
logger = logging.getLogger("faceguard-api")


@app.on_event("startup")
def warn_if_mock_mode_enabled() -> None:
  if settings.ai_mock_mode:
    logger.warning(
      "AI_MOCK_MODE=true is enabled. Identity and liveness verification are mocked; "
      "use this only for localhost integration testing."
    )


def failure_response(
  reason: str,
  *,
  liveness_passed: bool = False,
  face_match_passed: bool = False,
  liveness_score: float = 0.0,
  face_match_score: float = 0.0,
  single_face_detected: bool = False,
  challenge_passed: bool = False,
  quality_checks: dict | None = None,
) -> VerifyResponse:
  return VerifyResponse(
    success=False,
    liveness_passed=liveness_passed,
    face_match_passed=face_match_passed,
    single_face_detected=single_face_detected,
    challenge_passed=challenge_passed,
    liveness_score=liveness_score,
    face_match_score=face_match_score,
    face_match_metric=settings.face_match_metric,
    liveness_threshold=settings.liveness_threshold,
    face_match_threshold=settings.face_match_threshold,
    failure_reason=reason,
    quality_checks=quality_checks if settings.ai_debug else None,
  )


@app.get("/health")
def health() -> dict:
  model_status = get_model_status()

  return {
    "success": True,
    "service": "faceguard-api",
    "mock_mode": settings.ai_mock_mode,
    "debug": settings.ai_debug,
    "face_match_metric": settings.face_match_metric,
    "quality_check_enabled": settings.quality_check_enabled,
    "service_ready": model_status["service_ready"],
    "models": {
      model_name: model["ready"]
      for model_name, model in model_status["models"].items()
    },
  }


@app.post(
  "/v1/verify",
  response_model=VerifyResponse,
  response_model_exclude_none=True,
  dependencies=[Depends(validate_ai_service_secret)],
)
async def verify(
  passport_photo: UploadFile = File(...),
  live_captured_frame: UploadFile = File(...),
  challenge_frames: List[UploadFile] = File(default=[]),
  challenge_metadata: str = Form(default="{}"),
) -> VerifyResponse:
  # Decode uploads before the model pipeline starts.
  passport_image = await read_upload_image(passport_photo, settings.max_image_mb)
  live_image = await read_upload_image(live_captured_frame, settings.max_image_mb)
  challenge_images = [
    await read_upload_image(frame, settings.max_image_mb) for frame in challenge_frames
  ]

  if settings.ai_mock_mode:
    return VerifyResponse(
      success=True,
      liveness_passed=True,
      face_match_passed=True,
      single_face_detected=True,
      challenge_passed=True,
      liveness_score=max(settings.liveness_threshold, 0.9),
      face_match_score=max(settings.face_match_threshold, 0.75),
      face_match_metric=settings.face_match_metric,
      liveness_threshold=settings.liveness_threshold,
      face_match_threshold=settings.face_match_threshold,
      failure_reason="",
    )

  try:
    detector = FaceDetector()
    liveness_detector = LivenessDetector()
    face_verifier = FaceVerifier()
    challenge_verifier = ChallengeVerifier(detector)
  except ModelMissingError as error:
    return failure_response(f"MODEL_FILE_MISSING: {error}")

  passport_faces = detector.detect(passport_image)
  live_faces = detector.detect(live_image)

  if not passport_faces:
    return failure_response("PASSPORT_FACE_NOT_FOUND")

  if not live_faces:
    return failure_response("LIVE_FACE_NOT_FOUND")

  if settings.require_single_face and len(passport_faces) != 1:
    return failure_response("MULTIPLE_FACES_DETECTED")

  if settings.require_single_face and len(live_faces) != 1:
    return failure_response("MULTIPLE_FACES_DETECTED")

  passport_face = passport_faces[0]
  live_face = live_faces[0]
  single_face_detected = len(passport_faces) == 1 and len(live_faces) == 1

  quality_checks = {}

  if settings.quality_check_enabled:
    # Reject weak frames early so the model scores stay easier to trust.
    passport_quality_ok, passport_quality_reason, passport_quality_checks = (
      validate_image_quality(
        passport_image,
        passport_face,
        "Passport-size photo",
        "PASSPORT",
      )
    )
    live_quality_ok, live_quality_reason, live_quality_checks = validate_image_quality(
      live_image,
      live_face,
      "Live captured frame",
      "LIVE",
    )
    quality_checks = {
      "passport_photo": passport_quality_checks,
      "live_captured_frame": live_quality_checks,
    }

    if not passport_quality_ok:
      return failure_response(
        passport_quality_reason,
        single_face_detected=single_face_detected,
        quality_checks=quality_checks,
      )

    if not live_quality_ok:
      return failure_response(
        live_quality_reason,
        single_face_detected=single_face_detected,
        quality_checks=quality_checks,
      )

  try:
    challenge_passed, challenge_reason = challenge_verifier.verify(
      challenge_images,
      challenge_metadata,
    )
    liveness_score = liveness_detector.score(live_image, live_face)
    face_match_score, face_match_metric = face_verifier.compare(
      passport_image,
      passport_face,
      live_image,
      live_face,
    )
  except Exception:
    logger.exception("Face verification model pipeline failed")
    return failure_response(
      "FACE_VERIFICATION_MODEL_ERROR",
      single_face_detected=single_face_detected,
      quality_checks=quality_checks,
    )

  liveness_passed = liveness_score >= settings.liveness_threshold
  face_match_passed = (
    face_match_score <= settings.face_match_threshold
    if face_match_metric == "l2"
    else face_match_score >= settings.face_match_threshold
  )
  # The final decision stays strict: every verification layer must agree.
  success = (
    liveness_passed
    and face_match_passed
    and single_face_detected
    and challenge_passed
  )

  failure_reasons = []

  if not liveness_passed:
    failure_reasons.append("LIVENESS_FAILED")

  if not face_match_passed:
    failure_reasons.append("FACE_MATCH_FAILED")

  if not challenge_passed:
    failure_reasons.append(challenge_reason or "CHALLENGE_FAILED")

  return VerifyResponse(
    success=success,
    liveness_passed=liveness_passed,
    face_match_passed=face_match_passed,
    single_face_detected=single_face_detected,
    challenge_passed=challenge_passed,
    liveness_score=liveness_score,
    face_match_score=face_match_score,
    face_match_metric=face_match_metric,
    liveness_threshold=settings.liveness_threshold,
    face_match_threshold=settings.face_match_threshold,
    failure_reason="; ".join(failure_reasons),
    quality_checks=quality_checks if settings.ai_debug else None,
  )
