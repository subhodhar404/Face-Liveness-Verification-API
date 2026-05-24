# FaceGuard API

A standalone FastAPI service for face verification, liveness scoring, and passport-photo to live-frame face matching. It is designed to sit behind your main backend as a private internal service.

The service accepts a passport-size photo, a live captured frame, and optional challenge frames. It returns a clean JSON decision with liveness status, face-match status, model scores, thresholds, and a safe failure reason.

## Features

- Face detection with OpenCV YuNet.
- Face matching with OpenCV SFace.
- Liveness scoring with a MiniFASNet/Silent-Face-Anti-Spoofing-style ONNX model.
- Image quality checks for face size, blur, and brightness.
- Optional challenge-frame validation for blink, head-turn, and smile flows.
- Shared-secret protection for backend-to-service calls.
- Mock mode for local frontend/backend integration before model files are ready.
- Clear health and model readiness checks.

## Project Structure

```txt
app/
  main.py             FastAPI routes and verification flow
  config.py           Environment-driven service settings
  face_detector.py    YuNet face detection wrapper
  face_verifier.py    SFace embedding comparison wrapper
  liveness.py         Liveness and challenge-frame checks
  quality.py          Face crop quality checks
  security.py         Internal shared-secret validation
  schemas.py          API response models
  check_models.py     Model readiness checker
models/
  README.md           Model placement instructions
requirements.txt      Python dependencies
```

## Requirements

- Python 3.10 or newer.
- OpenCV-compatible CPU runtime.
- Three local model files in `models/`.
- A backend server that can send multipart form-data.

Required model files:

```txt
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
models/minifasnet_liveness.onnx
```

Model binaries are not committed to this repository. Keep them local, mount them in Docker, or copy them during deployment.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/faceguard-api.git
cd faceguard-api

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Then place the model files in `models/`.

## Environment Variables

```env
AI_SERVICE_SECRET=dev-secret-change-in-production
YUNET_MODEL_FILE=face_detection_yunet_2023mar.onnx
SFACE_MODEL_FILE=face_recognition_sface_2021dec.onnx
LIVENESS_MODEL_FILE=minifasnet_liveness.onnx
LIVENESS_THRESHOLD=0.80
FACE_MATCH_THRESHOLD=0.60
FACE_MATCH_METRIC=cosine
AI_MOCK_MODE=false
AI_DEBUG=false
MAX_IMAGE_MB=5
MIN_FACE_SIZE=80
REQUIRE_SINGLE_FACE=true
QUALITY_CHECK_ENABLED=true
BLUR_THRESHOLD=35
MIN_BRIGHTNESS=35
MAX_BRIGHTNESS=225
```

Use a strong random `AI_SERVICE_SECRET` in production. The same value must be configured in the backend that calls this service.

## Run Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Health check:

```bash
curl http://localhost:8001/health
```

Model readiness check:

```bash
python -m app.check_models --strict
```

`service_ready` is `true` only when `AI_MOCK_MODE=false` and all required model files exist with non-zero size.

## Verify API

```http
POST /v1/verify
X-AI-Service-Secret: your-shared-secret
Content-Type: multipart/form-data
```

Multipart fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `passport_photo` | file | yes | Passport-size or profile photo to compare against. |
| `live_captured_frame` | file | yes | Best live camera frame for matching and liveness. |
| `challenge_frames` | file[] | no | Frames captured while the user completes prompts. |
| `challenge_metadata` | string | no | JSON string describing expected and completed prompts. |

Supported image types:

```txt
image/jpeg
image/jpg
image/png
image/webp
```

Example request:

```bash
curl -X POST http://localhost:8001/v1/verify \
  -H "X-AI-Service-Secret: dev-secret-change-in-production" \
  -F "passport_photo=@passport.jpg" \
  -F "live_captured_frame=@live.jpg" \
  -F "challenge_frames=@challenge-1.jpg" \
  -F "challenge_frames=@challenge-2.jpg" \
  -F "challenge_metadata={\"challenges\":[\"blink\",\"turn_left\",\"turn_right\",\"smile\"]}"
```

Successful response:

```json
{
  "success": true,
  "liveness_passed": true,
  "face_match_passed": true,
  "single_face_detected": true,
  "challenge_passed": true,
  "liveness_score": 0.91,
  "face_match_score": 0.74,
  "face_match_metric": "cosine",
  "liveness_threshold": 0.8,
  "face_match_threshold": 0.6,
  "failure_reason": "",
  "provider": "opencv-yunet-sface-minifasnet"
}
```

Failed response:

```json
{
  "success": false,
  "liveness_passed": true,
  "face_match_passed": false,
  "single_face_detected": true,
  "challenge_passed": true,
  "liveness_score": 0.88,
  "face_match_score": 0.42,
  "face_match_metric": "cosine",
  "liveness_threshold": 0.8,
  "face_match_threshold": 0.6,
  "failure_reason": "FACE_MATCH_FAILED",
  "provider": "opencv-yunet-sface-minifasnet"
}
```

## Backend Integration

Your public frontend should call your own backend. Your backend should call this AI service privately.

Example Node.js integration:

```js
const verifyFace = async ({
  passportPhoto,
  liveCapturedFrame,
  challengeFrames = [],
  challengeMetadata = {}
}) => {
  const formData = new FormData();

  formData.append(
    'passport_photo',
    new Blob([passportPhoto.buffer], { type: passportPhoto.mimetype }),
    passportPhoto.originalname || 'passport.jpg'
  );
  formData.append(
    'live_captured_frame',
    new Blob([liveCapturedFrame.buffer], { type: liveCapturedFrame.mimetype }),
    liveCapturedFrame.originalname || 'live.jpg'
  );

  challengeFrames.forEach((frame, index) => {
    formData.append(
      'challenge_frames',
      new Blob([frame.buffer], { type: frame.mimetype }),
      frame.originalname || `challenge-${index}.jpg`
    );
  });

  formData.append('challenge_metadata', JSON.stringify(challengeMetadata));

  const response = await fetch(`${process.env.AI_SERVICE_URL}/v1/verify`, {
    method: 'POST',
    headers: {
      'X-AI-Service-Secret': process.env.AI_SERVICE_SECRET
    },
    body: formData
  });

  if (!response.ok) {
    throw new Error(`Face verification failed with status ${response.status}`);
  }

  return response.json();
};
```

Recommended backend environment variables:

```env
AI_SERVICE_URL=http://localhost:8001
AI_SERVICE_SECRET=the-same-secret-used-by-faceguard-api
AI_SERVICE_TIMEOUT_MS=30000
```

## Challenge Metadata

The service accepts flexible metadata, but this shape is recommended:

```json
{
  "challenges": ["blink", "turn_left", "turn_right", "smile"],
  "challengeSequence": ["blink", "turn_left", "turn_right", "smile"],
  "completedChallengeSequence": ["blink", "turn_left", "turn_right", "smile"],
  "capturedFrameCount": 8,
  "detector": "mediapipe_face_landmarker"
}
```

`challenge_frames` should contain enough frames to prove that the user stayed visible during the prompt flow. The service checks that the frames consistently contain exactly one face.

## Failure Reasons

Common `failure_reason` values:

| Reason | Meaning |
| --- | --- |
| `PASSPORT_FACE_NOT_FOUND` | No face was detected in the passport photo. |
| `LIVE_FACE_NOT_FOUND` | No face was detected in the live frame. |
| `MULTIPLE_FACES_DETECTED` | More than one face was detected when single-face mode is enabled. |
| `FACE_TOO_SMALL` | The detected face is smaller than `MIN_FACE_SIZE`. |
| `LIVE_FRAME_TOO_BLURRY` | The live face crop did not pass blur checks. |
| `LIVE_FRAME_TOO_DARK` | The live face crop is underexposed. |
| `LIVE_FRAME_TOO_BRIGHT` | The live face crop is overexposed. |
| `PASSPORT_FRAME_TOO_BLURRY` | The passport face crop did not pass blur checks. |
| `LIVENESS_FAILED` | The liveness score is below `LIVENESS_THRESHOLD`. |
| `FACE_MATCH_FAILED` | The two faces did not meet the configured match threshold. |
| `MODEL_FILE_MISSING` | One or more required model files are missing. |
| `FACE_VERIFICATION_MODEL_ERROR` | The model pipeline failed unexpectedly. |

## Mock Mode

`AI_MOCK_MODE=true` returns a passing verification response without running real models. Use it only for local integration testing when your frontend and backend flow are not ready for real model files yet.

Never enable mock mode for production or real identity verification.

## Security Notes

- Keep this service behind your backend or a private network.
- Do not expose `/v1/verify` directly to browsers.
- Use HTTPS in production.
- Store `AI_SERVICE_SECRET` as a real secret, not in source control.
- Keep `AI_DEBUG=false` in production.
- Do not log raw images, embeddings, access tokens, or secrets.

## Tuning

`FACE_MATCH_THRESHOLD=0.60` with `FACE_MATCH_METRIC=cosine` is a starting point. Tune it with your own camera quality, passport-photo quality, and risk tolerance.

For stricter matching, increase the cosine threshold. For `l2`, lower values are stricter and the pass condition becomes `score <= FACE_MATCH_THRESHOLD`.

## License

MIT License. See [LICENSE](LICENSE).
