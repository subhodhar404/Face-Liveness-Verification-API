from fastapi import Header, HTTPException, status

from .config import settings


def validate_ai_service_secret(
  x_ai_service_secret: str | None = Header(default=None, alias="X-AI-Service-Secret")
) -> None:
  if not settings.ai_service_secret:
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail="AI service secret is not configured",
    )

  if x_ai_service_secret != settings.ai_service_secret:
    raise HTTPException(
      status_code=status.HTTP_403_FORBIDDEN,
      detail="Invalid AI service secret",
    )
