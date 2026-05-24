import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

load_dotenv(BASE_DIR / ".env")


def _as_bool(value: str, default: bool = False) -> bool:
  if value is None:
    return default

  return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str, default: float) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def _as_int(value: str, default: int) -> int:
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _metric(value: str) -> str:
  normalized = str(value or "cosine").strip().lower()
  return normalized if normalized in {"cosine", "l2"} else "cosine"


class Settings:
  ai_service_secret = os.getenv("AI_SERVICE_SECRET", "")
  liveness_threshold = _as_float(os.getenv("LIVENESS_THRESHOLD"), 0.80)
  face_match_threshold = _as_float(os.getenv("FACE_MATCH_THRESHOLD"), 0.60)
  face_match_metric = _metric(os.getenv("FACE_MATCH_METRIC"))
  ai_mock_mode = _as_bool(os.getenv("AI_MOCK_MODE"), False)
  ai_debug = _as_bool(os.getenv("AI_DEBUG"), False)
  max_image_mb = _as_int(os.getenv("MAX_IMAGE_MB"), 5)
  min_face_size = _as_int(os.getenv("MIN_FACE_SIZE"), 80)
  require_single_face = _as_bool(os.getenv("REQUIRE_SINGLE_FACE"), True)
  quality_check_enabled = _as_bool(os.getenv("QUALITY_CHECK_ENABLED"), True)
  blur_threshold = _as_float(os.getenv("BLUR_THRESHOLD"), 35.0)
  min_brightness = _as_float(os.getenv("MIN_BRIGHTNESS"), 35.0)
  max_brightness = _as_float(os.getenv("MAX_BRIGHTNESS"), 225.0)

  # Model binaries stay outside git so deployments can bring their own weights.
  yunet_model_path = MODELS_DIR / os.getenv(
    "YUNET_MODEL_FILE", "face_detection_yunet_2023mar.onnx"
  )
  sface_model_path = MODELS_DIR / os.getenv(
    "SFACE_MODEL_FILE", "face_recognition_sface_2021dec.onnx"
  )
  liveness_model_path = MODELS_DIR / os.getenv(
    "LIVENESS_MODEL_FILE", "minifasnet_liveness.onnx"
  )


settings = Settings()
