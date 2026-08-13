"""Download the CRADLEBench and DeepSuiMind datasets from the HuggingFace Hub.

Both are public (no gating). Files land under data/raw/ and are gitignored.

Usage:
    python src/download_data.py
"""
from pathlib import Path
from huggingface_hub import snapshot_download

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# (repo_id, repo_type, local subdir)
SOURCES = [
    # CRADLE Bench — clinician-annotated crisis/safety-risk benchmark (arXiv 2510.23845)
    ("SungJoo/Cradle-Bench", "dataset", "cradlebench"),
    # DeepSuiMind — implicit suicidal ideation dialogues (arXiv 2502.17899, HF footnote)
    #("babytreecc/Implicit-suicide-detection", "dataset", "deepsuimind"), ##no longer using this dataset
    # CRADLE-Dialogue - clinician-annotated crisis/safety-risk multi-turn benchmark
    ("SungJoo/Cradle-Dialogue", "dataset", "cradledialogue")
]


def main() -> None:
    for repo_id, repo_type, subdir in SOURCES:
        out = RAW / subdir
        out.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(out))
        print(f"OK  {repo_id}  ->  {path}")


if __name__ == "__main__":
    main()
