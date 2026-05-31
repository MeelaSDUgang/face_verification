import os
import uuid
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .verifier import FaceVerifier
from .store import EmbeddingStore
from .config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

verifier: FaceVerifier = None
store: EmbeddingStore = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global verifier, store
    log.info("Loading face model (%s)…", settings.model_name)
    verifier = FaceVerifier(
        model_name=settings.model_name,
        detector=settings.detector_backend,
        threshold=settings.similarity_threshold,
    )
    store = EmbeddingStore(db_path=settings.db_path)
    log.info("Service ready.")
    yield
    store.close()

app = FastAPI(
    title="Face Verification Service",
    version="1.0.0",
    description="Local face verification microservice — register faces, verify identity.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Api-Key header.")
    return x_api_key

class RegisterResponse(BaseModel):
    success: bool
    user_id: str
    message: str

class VerifyResponse(BaseModel):
    success: bool
    verified: bool
    user_id: str
    confidence: float       
    distance: float
    threshold: float
    elapsed_ms: float

class DeleteResponse(BaseModel):
    success: bool
    user_id: str
    message: str

class HealthResponse(BaseModel):
    status: str
    model: str
    detector: str
    registered_users: int

async def save_upload(upload: UploadFile) -> Path:
    """Save uploaded file to a temp path; caller must delete."""
    suffix = Path(upload.filename).suffix if upload.filename else ".jpg"
    tmp = Path("/tmp") / f"{uuid.uuid4()}{suffix}"
    content = await upload.read()
    tmp.write_bytes(content)
    return tmp


@app.get("/health", response_model=HealthResponse)
def health(_: str = Depends(require_api_key)):
    return HealthResponse(
        status="ok",
        model=settings.model_name,
        detector=settings.detector_backend,
        registered_users=store.count(),
    )


@app.post("/register/{user_id}", response_model=RegisterResponse)
async def register(
    user_id: str,
    file: UploadFile = File(..., description="Clear front-facing photo (JPEG/PNG)"),
    _: str = Depends(require_api_key),
):
    tmp = await save_upload(file)
    try:
        embedding = verifier.embed(str(tmp))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        tmp.unlink(missing_ok=True)

    store.upsert(user_id, embedding)
    return RegisterResponse(success=True, user_id=user_id, message="Face registered.")


@app.post("/verify/{user_id}", response_model=VerifyResponse)
async def verify(
    user_id: str,
    file: UploadFile = File(..., description="Live photo to verify against registered face"),
    _: str = Depends(require_api_key),
):
    ref_embedding = store.get(user_id)
    if ref_embedding is None:
        raise HTTPException(status_code=404, detail=f"No registered face for user '{user_id}'.")

    tmp = await save_upload(file)
    t0 = time.perf_counter()
    try:
        result = verifier.compare(ref_embedding, str(tmp))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        tmp.unlink(missing_ok=True)

    elapsed = (time.perf_counter() - t0) * 1000
    return VerifyResponse(
        success=True,
        verified=result["verified"],
        user_id=user_id,
        confidence=result["confidence"],
        distance=result["distance"],
        threshold=result["threshold"],
        elapsed_ms=round(elapsed, 1),
    )


@app.delete("/users/{user_id}", response_model=DeleteResponse)
def delete_user(user_id: str, _: str = Depends(require_api_key)):
    deleted = store.delete(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    return DeleteResponse(success=True, user_id=user_id, message="Face data deleted.")


@app.get("/users", response_model=list[str])
def list_users(_: str = Depends(require_api_key)):
    return store.list_users()
