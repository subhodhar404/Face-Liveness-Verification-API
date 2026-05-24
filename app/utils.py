from typing import Iterable

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


async def read_upload_image(file: UploadFile, max_image_mb: int) -> np.ndarray:
  if file.content_type not in ALLOWED_IMAGE_TYPES:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail=f"Unsupported image type for {file.filename or 'upload'}",
    )

  max_bytes = max_image_mb * 1024 * 1024
  contents = await file.read()

  if not contents:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail=f"Empty image upload for {file.filename or 'upload'}",
    )

  if len(contents) > max_bytes:
    raise HTTPException(
      status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
      detail=f"Image upload exceeds {max_image_mb}MB",
    )

  image_array = np.frombuffer(contents, dtype=np.uint8)
  image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

  if image is None:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail=f"Could not decode image {file.filename or 'upload'}",
    )

  return image


def softmax(values: Iterable[float]) -> np.ndarray:
  array = np.asarray(list(values), dtype=np.float32)
  shifted = array - np.max(array)
  exp_values = np.exp(shifted)
  return exp_values / np.sum(exp_values)


def clamp_crop(image: np.ndarray, box: list[float]) -> np.ndarray:
  height, width = image.shape[:2]
  x, y, w, h = [int(round(value)) for value in box]
  x1 = max(0, x)
  y1 = max(0, y)
  x2 = min(width, x + w)
  y2 = min(height, y + h)

  if x2 <= x1 or y2 <= y1:
    return image

  return image[y1:y2, x1:x2]
