"""
Offline Training Script — Item-Based Collaborative Filtering
=============================================================

Builds the item-item similarity matrix from user-rating data and persists
the resulting artifacts to disk so the FastAPI backend can load them on
startup without recalculating.

Usage:
    python -m backend.app.models.train_collaborative

Artifacts produced (in ``data/processed/collaborative_artifacts/``):
    - ``item_similarity_matrix.joblib``  — dense float32 cosine-similarity matrix
    - ``movie_index_mapping.joblib``     — bidirectional movieId ↔ matrix-row mapping
"""

import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "collabrative_features.csv"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "processed" / "collaborative_artifacts"


# ──────────────────────────────────────────────────────────────────────────
# Step 1 — Load CSV
# ──────────────────────────────────────────────────────────────────────────

def load_ratings(path: Path) -> pd.DataFrame:
    """
    Load the ratings CSV with memory-optimised dtypes.

    Using ``int32`` for IDs and ``float32`` for ratings reduces the
    DataFrame footprint by roughly 50 % compared to the default int64/float64.
    """
    logger.info("Loading ratings from %s …", path)

    if not path.exists():
        raise FileNotFoundError(f"Ratings file not found: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "normalized_rating": np.float32,
        },
    )
    logger.info(
        "Loaded %s ratings  (%d unique users, %d unique movies)",
        f"{len(df):,}",
        df["userId"].nunique(),
        df["movieId"].nunique(),
    )
    return df


# ──────────────────────────────────────────────────────────────────────────
# Step 2 — Build sparse User-Item matrix
# ──────────────────────────────────────────────────────────────────────────

def build_sparse_matrix(df: pd.DataFrame) -> tuple[csr_matrix, dict]:
    """
    Build a sparse Item × User matrix in CSR format.

    Rows  = movies  (items)
    Cols  = users
    Values = normalized_rating

    Returns:
        sparse_matrix: CSR matrix of shape (num_movies, num_users)
        mapping:       dict with keys ``movie_id_to_idx`` and ``idx_to_movie_id``
    """
    logger.info("Building sparse Item × User matrix …")

    # Factorize IDs → contiguous 0-based indices
    movie_codes, movie_uniques = pd.factorize(df["movieId"], sort=True)
    user_codes, _user_uniques = pd.factorize(df["userId"], sort=True)

    # COO → CSR  (COO is the fastest way to construct a sparse matrix)
    sparse_matrix = coo_matrix(
        (df["normalized_rating"].values, (movie_codes, user_codes)),
        shape=(len(movie_uniques), len(_user_uniques)),
        dtype=np.float32,
    ).tocsr()

    # Bidirectional mapping for movieId ↔ matrix row index
    mapping = {
        "movie_id_to_idx": {int(mid): int(idx) for idx, mid in enumerate(movie_uniques)},
        "idx_to_movie_id": {int(idx): int(mid) for idx, mid in enumerate(movie_uniques)},
    }

    logger.info(
        "Sparse matrix shape: %s  |  nnz: %s  |  density: %.4f%%",
        sparse_matrix.shape,
        f"{sparse_matrix.nnz:,}",
        100.0 * sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]),
    )
    return sparse_matrix, mapping


# ──────────────────────────────────────────────────────────────────────────
# Step 3 — Compute cosine similarity
# ──────────────────────────────────────────────────────────────────────────

def compute_similarity(sparse_matrix: csr_matrix) -> np.ndarray:
    """
    Compute item-item cosine similarity.

    ``sklearn.metrics.pairwise.cosine_similarity`` is optimised for sparse
    input and parallelises internally via BLAS.  The output is a dense
    (num_movies × num_movies) float32 array.
    """
    logger.info("Computing cosine similarity (this may take a while) …")

    similarity = cosine_similarity(sparse_matrix, dense_output=True)

    # Down-cast to float32 to halve memory usage on disk and at load time
    similarity = similarity.astype(np.float32)

    logger.info(
        "Similarity matrix shape: %s  |  size: %.1f MB",
        similarity.shape,
        similarity.nbytes / (1024 ** 2),
    )
    return similarity


# ──────────────────────────────────────────────────────────────────────────
# Step 4 — Save artifacts
# ──────────────────────────────────────────────────────────────────────────

def save_artifacts(
    similarity: np.ndarray,
    mapping: dict,
    output_dir: Path,
) -> None:
    """
    Serialize the similarity matrix and index mapping to disk using joblib.

    ``compress=3`` gives a good balance between file size and
    serialization speed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_path = output_dir / "item_similarity_matrix.joblib"
    map_path = output_dir / "movie_index_mapping.joblib"

    logger.info("Saving similarity matrix to %s …", sim_path)
    joblib.dump(similarity, sim_path, compress=3)

    logger.info("Saving index mapping to %s …", map_path)
    joblib.dump(mapping, map_path, compress=3)

    # Report file sizes
    sim_size = sim_path.stat().st_size / (1024 ** 2)
    map_size = map_path.stat().st_size / (1024 ** 2)
    logger.info(
        "Artifacts saved — similarity: %.1f MB, mapping: %.2f MB",
        sim_size,
        map_size,
    )


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full training pipeline."""
    overall_start = time.perf_counter()

    # 1. Load
    df = load_ratings(DATA_PATH)

    # 2. Sparse matrix
    sparse_matrix, mapping = build_sparse_matrix(df)

    # Free the DataFrame — no longer needed
    del df

    # 3. Similarity
    similarity = compute_similarity(sparse_matrix)

    # Free the sparse matrix — no longer needed
    del sparse_matrix

    # 4. Save
    save_artifacts(similarity, mapping, ARTIFACT_DIR)

    elapsed = time.perf_counter() - overall_start
    logger.info("✅  Training complete in %.1f s", elapsed)


if __name__ == "__main__":
    main()
