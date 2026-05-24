import cv2

from .config import settings


def validate_face_quality(face: dict) -> tuple[bool, str]:
  x, y, width, height = face["box"]
  minimum_size = settings.min_face_size

  if width < minimum_size or height < minimum_size:
    return False, "FACE_TOO_SMALL"

  return True, ""


def validate_image_quality(
  image,
  face: dict,
  label: str,
  frame_prefix: str,
) -> tuple[bool, str, dict]:
  x, y, width, height = [int(round(value)) for value in face["box"]]
  checks = {
    "label": label,
    "face_width": width,
    "face_height": height,
    "min_face_size": settings.min_face_size,
  }
  face_ok, face_reason = validate_face_quality(face)

  if not face_ok:
    return False, face_reason, checks

  image_height, image_width = image.shape[:2]
  x1 = max(0, x)
  y1 = max(0, y)
  x2 = min(image_width, x + width)
  y2 = min(image_height, y + height)

  if x2 <= x1 or y2 <= y1:
    return False, f"{frame_prefix}_FACE_CROP_INVALID", checks

  crop = image[y1:y2, x1:x2]
  gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
  brightness = float(gray.mean())
  blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
  checks.update(
    {
      "brightness": round(brightness, 2),
      "blur_score": round(blur_score, 2),
      "min_brightness": settings.min_brightness,
      "max_brightness": settings.max_brightness,
      "blur_threshold": settings.blur_threshold,
    }
  )

  if brightness < settings.min_brightness:
    return False, f"{frame_prefix}_FRAME_TOO_DARK", checks

  if brightness > settings.max_brightness:
    return False, f"{frame_prefix}_FRAME_TOO_BRIGHT", checks

  if blur_score < settings.blur_threshold:
    return False, f"{frame_prefix}_FRAME_TOO_BLURRY", checks

  return True, "", checks
