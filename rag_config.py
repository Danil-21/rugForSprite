from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
VECTOR_DB_DIR = BASE_DIR / "data" / "chroma_db"


__all__ = [
    "BASE_DIR",
    "DATA_RAW_DIR",
    "VECTOR_DB_DIR",
]
