from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    model_name: str = "Facenet512"      
    detector_backend: str = "retinaface"
    similarity_threshold: float = 0.40     

    db_path: str = "data/embeddings.db"

    api_key: Optional[str] = None         

    allowed_origins: list[str] = ["*"]

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1                      

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
