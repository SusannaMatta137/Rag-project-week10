# Make the RAG app importable when pytest is run from the repo root.
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "student-rag-project-main"
sys.path.insert(0, str(APP_DIR))
