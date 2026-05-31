import numpy as np
from typing import Any

try:
    from deepface import DeepFace
except ImportError as e:
    raise ImportError(
        "DeepFace is not installed. Run:  pip install deepface tf-keras\n"
        f"Original error: {e}"
    )


class FaceVerifier:
    def __init__(
        self,
        model_name: str = "Facenet512",
        detector: str = "retinaface",
        threshold: float = 0.40,
        distance_metric: str = "cosine",
    ):
        self.model_name = model_name
        self.detector = detector
        self.threshold = threshold
        self.distance_metric = distance_metric

        self._warmup()

    def _warmup(self):

        try:
            DeepFace.build_model(self.model_name)
        except Exception:
            pass 

    def embed(self, image_path: str) -> np.ndarray:
        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector,
                enforce_detection=True,
                align=True,
            )
        except ValueError as e:
            raise ValueError(f"No face detected in image: {e}")
        except Exception as e:
            raise ValueError(f"Face embedding failed: {e}")

        if not result:
            raise ValueError("No face detected in image.")

        best = max(result, key=lambda r: r.get("facial_area", {}).get("w", 0) * r.get("facial_area", {}).get("h", 0))
        return np.array(best["embedding"], dtype=np.float32)

    def compare(self, reference_embedding: np.ndarray, live_image_path: str) -> dict[str, Any]:
        live_embedding = self.embed(live_image_path)
        distance = self._cosine_distance(reference_embedding, live_embedding)
        verified = distance <= self.threshold

        confidence = float(max(0.0, 1.0 - distance / max(self.threshold * 2, 1e-6)))
        return {
            "verified": verified,
            "confidence": round(confidence, 4),
            "distance": round(float(distance), 4),
            "threshold": self.threshold,
        }

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-10)
        b = b / (np.linalg.norm(b) + 1e-10)
        return float(1.0 - np.dot(a, b))
