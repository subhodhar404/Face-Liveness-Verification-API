from __future__ import annotations

import argparse
from pathlib import Path

from .config import settings


MODEL_FILES = {
  "yunet": settings.yunet_model_path,
  "sface": settings.sface_model_path,
  "liveness": settings.liveness_model_path,
}


def _file_status(path: Path) -> dict:
  exists = path.exists()
  size_bytes = path.stat().st_size if exists else 0

  return {
    "exists": exists,
    "size_bytes": size_bytes,
    "ready": exists and size_bytes > 0,
    "path": str(path),
  }


def get_model_status() -> dict:
  models = {
    name: _file_status(path)
    for name, path in MODEL_FILES.items()
  }
  all_models_ready = all(model["ready"] for model in models.values())
  service_ready = (not settings.ai_mock_mode) and all_models_ready

  return {
    "mock_mode": settings.ai_mock_mode,
    "service_ready": service_ready,
    "models": models,
  }


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Check Smart NID face verification model files."
  )
  parser.add_argument(
    "--strict",
    action="store_true",
    help="Exit with status 1 when real verification is not ready.",
  )
  args = parser.parse_args()
  status = get_model_status()

  print("Smart NID face verification model check")
  print(f"AI_MOCK_MODE={str(status['mock_mode']).lower()}")
  print(f"service_ready={str(status['service_ready']).lower()}")

  for model_name, model_status in status["models"].items():
    print(
      f"{model_name}: exists={str(model_status['exists']).lower()} "
      f"size_bytes={model_status['size_bytes']} path={model_status['path']}"
    )

  if args.strict and not status["service_ready"]:
    return 1

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
