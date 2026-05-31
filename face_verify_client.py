
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VerifyResult:
    verified: bool
    confidence: float       # 0–1
    distance: float
    threshold: float
    elapsed_ms: float

    def __str__(self):
        mark = "✓" if self.verified else "✗"
        return (
            f"{mark} verified={self.verified}  "
            f"confidence={self.confidence:.1%}  "
            f"distance={self.distance:.4f}  "
            f"({self.elapsed_ms:.0f} ms)"
        )


class FaceVerifyClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-Api-Key"] = api_key

    def register(self, user_id: str, image_path: str | Path) -> dict:
        """Register a reference face. Call once per user (or to update)."""
        with open(image_path, "rb") as f:
            r = self.session.post(
                f"{self.base_url}/register/{user_id}",
                files={"file": (Path(image_path).name, f, "image/jpeg")},
            )
        r.raise_for_status()
        return r.json()

    def verify(self, user_id: str, image_path: str | Path) -> VerifyResult:
        """Verify a live photo against the registered face."""
        with open(image_path, "rb") as f:
            r = self.session.post(
                f"{self.base_url}/verify/{user_id}",
                files={"file": (Path(image_path).name, f, "image/jpeg")},
            )
        r.raise_for_status()
        d = r.json()
        return VerifyResult(
            verified=d["verified"],
            confidence=d["confidence"],
            distance=d["distance"],
            threshold=d["threshold"],
            elapsed_ms=d["elapsed_ms"],
        )

    def delete(self, user_id: str) -> dict:
        """Remove a user's stored face data."""
        r = self.session.delete(f"{self.base_url}/users/{user_id}")
        r.raise_for_status()
        return r.json()

    def list_users(self) -> list[str]:
        r = self.session.get(f"{self.base_url}/users")
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        r = self.session.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    client = FaceVerifyClient(url)
    print("Health:", client.health())
    print("Users:", client.list_users())
