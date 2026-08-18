import sys
from pathlib import Path

# Proje kökünü Python path'ine ekle — RAG, backend gibi paketler bulunabilsin
sys.path.insert(0, str(Path(__file__).parent))
