"""Download pretrained model artifacts and optional GloVe embeddings.

Fetches the trained Module 1, Module 3, and Module 4 artifacts hosted on the
project's Hugging Face Space (benzmehdi/nlppro) so a fresh GitHub clone can
run the backend immediately without storing large binaries in Git.
"""

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download

ROOT_DIR = Path(__file__).resolve().parent.parent
SPACE_REPO_ID = "benzmehdi/nlppro"

SPACE_ARTIFACTS = [
    "module 1 models/bm25.pkl",
    "module 1 models/corpus_index.json",
    "module 1 models/tfidf_matrix.npz",
    "module 1 models/tfidf_vectorizer.pkl",
    "module 3 model/qa_best.pt",
    "module 4 distilbert/config.json",
    "module 4 distilbert/model.safetensors",
    "module 4 distilbert/tokenizer.json",
    "module 4 distilbert/tokenizer_config.json",
    "summ_artifacts/config.json",
]

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


def download_hf_artifacts(force: bool = False) -> None:
    """Download model artifacts from Hugging Face Space into project folders."""
    for rel_path in SPACE_ARTIFACTS:
        target = ROOT_DIR / rel_path
        if target.exists() and not force:
            print(f"[skip] {rel_path} already exists")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] {rel_path} ...")
        hf_hub_download(
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            filename=rel_path,
            local_dir=str(ROOT_DIR),
        )
        print(f"[ok]    {rel_path}")


def download_glove(force: bool = False) -> None:
    """Download GloVe 6B 100d embeddings (~822 MB zip) for BiLSTM+BiDAF vocab."""
    target = ROOT_DIR / "glove.6B.100d.txt"
    if target.exists() and not force:
        print("[skip] glove.6B.100d.txt already exists")
        return

    zip_path = ROOT_DIR / "glove.6B.zip"
    print(f"[fetch] Downloading {GLOVE_URL} (~822 MB)...")
    urlretrieve(GLOVE_URL, zip_path)
    print("[extract] Extracting glove.6B.100d.txt ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("glove.6B.100d.txt") as src, open(target, "wb") as dst:
            dst.write(src.read())
    zip_path.unlink(missing_ok=True)
    print("[ok]    glove.6B.100d.txt ready")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NLP Pipeline model artifacts.")
    parser.add_argument("--with-glove", action="store_true", help="Also download GloVe 6B 100d embeddings (~822 MB).")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    args = parser.parse_args()

    download_hf_artifacts(force=args.force)
    if args.with_glove:
        download_glove(force=args.force)
    print("\nAll requested artifacts are ready.")


if __name__ == "__main__":
    main()
