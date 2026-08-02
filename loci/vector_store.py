"""
Runtime loader for precomputed brand embeddings.

Everything here is LOAD, never GENERATE: vectors, FAISS indices and the fitted
input encoder are all read from a folder that
vector_generation/generate_embeddings.py already wrote to disk. A store is
built once per brand — at API startup, or once in a script — and then held in
memory (encoder + vectors + FAISS indices) until the process exits. No
brand/generic text is ever embedded here; only the single new input text is,
later, via self.encoder inside DualCentroidModel.score().
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .fingerprint import BrandFingerprint, Chunk, GenericCorpus, Layer
from .input_encoder import InputEncoder

MIN_CALIB_WORDS = 8  # mirrors DualCentroidModel.MIN_CALIB_WORDS


class BrandVectorStore:
    def __init__(self, folder: Path):
        self.folder = folder
        self.manifest: dict = json.loads((folder / "manifest.json").read_text())

        self.brand_vectors = np.load(folder / "brand_vectors.npy")
        self.generic_vectors = np.load(folder / "generic_vectors.npy")
        self.brand_meta: list[dict] = json.loads((folder / "brand_meta.json").read_text())
        self.generic_meta: list[dict] = json.loads((folder / "generic_meta.json").read_text())

        # The FAISS vector base: brand + generic corpora, mounted once and
        # queried per request for nearest-neighbour evidence (see
        # nearest_brand_chunks / nearest_generic_chunks below).
        self.brand_index = faiss.read_index(str(folder / "brand.index"))
        self.generic_index = faiss.read_index(str(folder / "generic.index"))

        self.encoder = InputEncoder.load(folder)

        self.fingerprint = BrandFingerprint(
            brand_id=self.manifest["brand_id"],
            brand_name=self.manifest["brand_name"],
            chunks=[Chunk(**m) for m in self.brand_meta],
        )
        self.generic = GenericCorpus(
            industry=self.manifest["industry"],
            chunks=[Chunk(**m) for m in self.generic_meta],
        )

        self.warnings: list[str] = []
        ok, msg = self.fingerprint.is_scorable()
        if not ok:
            self.warnings.append(msg)

    @classmethod
    def load(cls, folder: Path | str) -> "BrandVectorStore":
        return cls(Path(folder))

    # ---------- per-layer vector slices, mirroring fingerprint.py's fallback rules ----------

    @staticmethod
    def _rows(meta: list[dict], layer: Layer) -> list[int]:
        return [i for i, m in enumerate(meta) if m["layer"] == layer.value]

    def brand_layer_vectors(self, layer: Layer) -> np.ndarray:
        idx = self._rows(self.brand_meta, layer)
        return self.brand_vectors[idx] if idx else self.brand_vectors[:0]

    def generic_layer_vectors(self, layer: Layer) -> np.ndarray:
        idx = self._rows(self.generic_meta, layer)
        # Generic corpora are thin per-layer; fall back to the whole corpus,
        # matching the original GenericCorpus.texts() behaviour.
        return self.generic_vectors[idx] if len(idx) >= 3 else self.generic_vectors

    @staticmethod
    def _calibration_subset(vectors: np.ndarray, meta: list[dict], idx: list[int]) -> np.ndarray:
        long_idx = [i for i in idx if meta[i]["words"] >= MIN_CALIB_WORDS]
        long_vecs = vectors[long_idx]
        return long_vecs if len(long_vecs) >= 2 else vectors[idx]

    def brand_calibration_vectors(self, layer: Layer) -> np.ndarray:
        idx = self._rows(self.brand_meta, layer)
        if not idx:
            return self.brand_vectors
        return self._calibration_subset(self.brand_vectors, self.brand_meta, idx)

    def generic_calibration_vectors(self, layer: Layer) -> np.ndarray:
        idx = self._rows(self.generic_meta, layer)
        if len(idx) < 3:
            idx = list(range(len(self.generic_meta)))
        return self._calibration_subset(self.generic_vectors, self.generic_meta, idx)

    # ---------- FAISS lookups: nearest evidence for a scored input ----------

    def nearest_brand_chunks(self, vec: np.ndarray, k: int = 3) -> list[str]:
        _, i = self.brand_index.search(
            vec.reshape(1, -1).astype("float32"), min(k, self.brand_index.ntotal))
        return [self.brand_meta[int(idx)]["text"] for idx in i[0] if idx != -1]

    def nearest_generic_chunks(self, vec: np.ndarray, k: int = 3) -> list[str]:
        _, i = self.generic_index.search(
            vec.reshape(1, -1).astype("float32"), min(k, self.generic_index.ntotal))
        return [self.generic_meta[int(idx)]["text"] for idx in i[0] if idx != -1]
