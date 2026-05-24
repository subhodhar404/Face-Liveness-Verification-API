# Model Files

Place the required model binaries in this folder before running real verification.

Required files:

- `face_detection_yunet_2023mar.onnx` for OpenCV YuNet face detection.
- `face_recognition_sface_2021dec.onnx` for OpenCV SFace matching.
- `minifasnet_liveness.onnx` for MiniFASNet/Silent-Face-Anti-Spoofing-style liveness detection.

Large model files are intentionally ignored by git. Keep this README in the folder and copy your own model files locally or during deployment.

Expected layout:

```txt
models/
  README.md
  face_detection_yunet_2023mar.onnx
  face_recognition_sface_2021dec.onnx
  minifasnet_liveness.onnx
```

Check readiness:

```bash
python -m app.check_models
```

`AI_MOCK_MODE=true` is only for localhost frontend/backend integration testing. It does not verify identity and can pass a different person's passport-size photo.

`AI_MOCK_MODE=false` requires the real model files above. In this mode the AI service detects exactly one face in the passport-size photo and live captured frame, runs liveness, and compares the two faces with OpenCV SFace. A different person's passport-size photo should fail with `FACE_MATCH_FAILED` when the cosine score is below `FACE_MATCH_THRESHOLD`, or when the L2 distance is above `FACE_MATCH_THRESHOLD`.

Use `FACE_MATCH_METRIC=cosine` or `FACE_MATCH_METRIC=l2` to match your threshold policy. Enable `AI_DEBUG=true` only in local development for safe score/quality metadata; it does not return raw images or embeddings.
