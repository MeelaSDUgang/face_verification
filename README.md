# 🔍 Face Verification Service

A self-hosted, local-inference face verification microservice built with **FastAPI + DeepFace**.  
No cloud, no third-party APIs, no biometric data leaving your server.

---

## Features

- ✅ Register a reference face per user (stores only the embedding vector, never the photo)
- ✅ Verify a live photo against the registered face — returns `verified` + confidence score
- ✅ REST API with automatic OpenAPI docs (`/docs`)
- ✅ SQLite storage — zero external dependencies
- ✅ Configurable model (`Facenet512`, `ArcFace`, `VGG-Face`, …)
- ✅ Optional API-key auth
- ✅ Docker-ready
- ✅ Drop-in Python + JS/TS client SDKs

---

## Quick Start

### Option A — Docker (recommended)

```bash
# 1. Clone / copy this folder
cd face-verify-service

# 2. (Optional) set an API key
cp .env.example .env
# Edit .env → API_KEY=your-secret

# 3. Build & run
docker compose up --build

# Service is live at http://localhost:8000
# Interactive API docs at http://localhost:8000/docs
```

### Option B — Direct Python

```bash
cd face-verify-service

# Python 3.10+ required
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

mkdir -p data
python run.py
```

> **First run** downloads model weights (~90 MB for Facenet512). Subsequent starts are instant.

---

## API Reference

All endpoints accept `X-Api-Key: <key>` header when `API_KEY` is set.

### `GET /health`
Returns service status and registered user count.

```json
{ "status": "ok", "model": "Facenet512", "detector": "retinaface", "registered_users": 3 }
```

---

### `POST /register/{user_id}`
Register a reference face.  
**Body:** `multipart/form-data` with field `file` (JPEG or PNG).

```bash
curl -X POST http://localhost:8000/register/alice \
  -F "file=@/path/to/alice_photo.jpg"
```

```json
{ "success": true, "user_id": "alice", "message": "Face registered." }
```

---

### `POST /verify/{user_id}`
Verify a live photo against the registered face.  
**Body:** `multipart/form-data` with field `file`.

```bash
curl -X POST http://localhost:8000/verify/alice \
  -F "file=@/path/to/live_photo.jpg"
```

```json
{
  "success": true,
  "verified": true,
  "user_id": "alice",
  "confidence": 0.923,
  "distance": 0.187,
  "threshold": 0.40,
  "elapsed_ms": 312.4
}
```

| Field | Meaning |
|---|---|
| `verified` | `true` if faces match |
| `confidence` | 0–1 score (1 = identical) |
| `distance` | Raw cosine distance (lower = more similar) |
| `threshold` | Configured match threshold |
| `elapsed_ms` | Inference time in milliseconds |

---

### `DELETE /users/{user_id}`
Remove a user's stored embedding.

```bash
curl -X DELETE http://localhost:8000/users/alice
```

---

### `GET /users`
List all registered user IDs.

---

## Integrating into Your App

### Python

Copy `face_verify_client.py` into your project:

```python
from face_verify_client import FaceVerifyClient

client = FaceVerifyClient("http://localhost:8000", api_key="your-key")

# During onboarding — register once
client.register("user_123", "/uploads/onboarding_photo.jpg")

# During login / sensitive action — verify
result = client.verify("user_123", "/tmp/webcam_capture.jpg")

if result.verified:
    print(f"✓ Identity confirmed ({result.confidence:.0%} confidence)")
    grant_access()
else:
    print(f"✗ Face mismatch (distance {result.distance:.3f})")
    deny_access()
```

---

### JavaScript / TypeScript

Copy `faceVerifyClient.ts` into your project:

```typescript
import { FaceVerifyClient } from './faceVerifyClient';

const client = new FaceVerifyClient({ baseUrl: 'http://localhost:8000', apiKey: 'your-key' });

// React example — file input
async function handleVerify(userId: string, file: File) {
  try {
    const result = await client.verify(userId, file);

    if (result.verified) {
      console.log(`✓ Confirmed — ${(result.confidence * 100).toFixed(0)}% confidence`);
    } else {
      console.log('✗ Face does not match');
    }
  } catch (err) {
    console.error('Verification error:', err);
  }
}
```

---

### React Hook (copy-paste ready)

```tsx
import { useState, useCallback } from 'react';
import { FaceVerifyClient, VerifyResult } from './faceVerifyClient';

const client = new FaceVerifyClient({ baseUrl: 'http://localhost:8000' });

export function useFaceVerify(userId: string) {
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState<VerifyResult | null>(null);
  const [error, setError]     = useState<string | null>(null);

  const verify = useCallback(async (file: File) => {
    setLoading(true); setError(null);
    try {
      const res = await client.verify(userId, file);
      setResult(res);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  return { verify, result, loading, error };
}

// Usage in a component:
// const { verify, result, loading } = useFaceVerify('user_123');
// <input type="file" onChange={e => verify(e.target.files![0])} />
// {result?.verified ? '✓ Match' : '✗ No match'}
```

---

### Webcam Capture (browser)

```javascript
// Capture a frame from webcam and send for verification
async function captureAndVerify(userId) {
  const stream  = await navigator.mediaDevices.getUserMedia({ video: true });
  const video   = document.createElement('video');
  const canvas  = document.createElement('canvas');
  video.srcObject = stream;
  await video.play();

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  stream.getTracks().forEach(t => t.stop());   // stop camera

  const blob   = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.92));
  const result = await client.verify(userId, blob);
  return result;
}
```

---

## Configuration

All settings can be set in `.env` or as environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Facenet512` | Face model (`VGG-Face`, `Facenet`, `Facenet512`, `ArcFace`) |
| `DETECTOR_BACKEND` | `retinaface` | Face detector (`retinaface`, `mtcnn`, `opencv`, `ssd`) |
| `SIMILARITY_THRESHOLD` | `0.40` | Cosine distance cutoff — lower = stricter |
| `DB_PATH` | `data/embeddings.db` | SQLite database file path |
| `API_KEY` | *(none)* | Set to require `X-Api-Key` on all requests |
| `PORT` | `8000` | Server port |

### Choosing the right model

| Model | Accuracy | Speed | Use case |
|---|---|---|---|
| `VGG-Face` | Good | Fast | Low-power servers, quick demos |
| `Facenet` | Better | Medium | Balanced default |
| `Facenet512` | Best | Medium | **Recommended** — high accuracy |
| `ArcFace` | Best | Slower | Maximum accuracy needed |

### Tuning the threshold

| Threshold | Behaviour |
|---|---|
| `0.30` | Very strict — fewer false accepts, more false rejects |
| `0.40` | **Balanced** (default) |
| `0.50` | More permissive — fewer false rejects |

---

## Running Tests

```bash
# Unit / integration tests (service must be running)
pip install pytest
pytest tests/ -v

# Test against a real face photo
TEST_PHOTO_PATH=/path/to/face.jpg pytest tests/ -v
```

---

## Production Checklist

- [ ] Set `API_KEY` in `.env`
- [ ] Restrict `ALLOWED_ORIGINS` to your domain
- [ ] Put behind HTTPS (nginx reverse proxy or Caddy)
- [ ] Mount `./data` as a persistent volume (Docker)
- [ ] Add liveness detection on the client side (blink challenge, head turn)
- [ ] Log verification events with user ID + timestamp (not the photo)
- [ ] Implement rate limiting (e.g. 5 attempts / user / minute)
- [ ] Get user consent for biometric data per your local regulations

---

## Project Structure

```
face-verify-service/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── verifier.py      # DeepFace wrapper — embedding + comparison
│   ├── store.py         # SQLite embedding storage
│   └── config.py        # Settings from env vars
├── tests/
│   └── test_service.py  # Integration tests
├── face_verify_client.py # Python SDK (copy into your project)
├── faceVerifyClient.ts   # JS/TS SDK (copy into your project)
├── run.py               # Server entrypoint
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```
