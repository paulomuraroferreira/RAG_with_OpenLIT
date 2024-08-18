from dataclasses import dataclass
from pathlib import Path

current_file_dir = Path(__file__).parent

@dataclass
class PathInfo:
    VECTOR_STORE_PATH:str = str(current_file_dir / "data" / "vector_store")
    PDFS_FOLDER_PATH:str = str(current_file_dir / "data" / "pdfs")
