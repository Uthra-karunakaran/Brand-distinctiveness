# Loci — Brand Distinctiveness Validation (prototype)

Answers two questions that most "brand voice" tools collapse into one:

- **Consistency** — *are you being you?* Measured against the brand's own fingerprint.
- **Distinctiveness** — *are you not sounding like everyone else?* Measured against a generic/competitor baseline.

They are computed from independent comparisons and combined only at the last step, into a 2×2.

```
python -m vector_generation.generate_embeddings   # offline: fit + embed once per brand
python demo.py                                    # worked example, four candidates, one per quadrant
uvicorn api:app --reload                          # service — loads precomputed embeddings at startup
```

---

## The three inputs

| # | Input | When | Required |
|---|-------|------|----------|
| 1 | **Brand assets** → Brand Fingerprint | once per brand (onboarding) | MVBF is the floor |
| 2 | **Competitor / industry corpus** → generic centroid | once per industry | yes — without it there is no distinctiveness axis |
| 3 | **New copy** | every request | yes |

### 1. Brand assets

`{asset_type: text | [texts]}`. Each asset type is mapped to a layer by `ASSET_LAYER_MAP` — that mapping is what turns scraped text into a structured fingerprint.

MVBF (the floor, enforced by `is_scorable()`): `name, tagline, mission, vision, values, about`.

### 2. Competitor corpus

Competitor homepages, product pages, industry-report language, templated marketing copy. 15–30 chunks is enough to form a usable centroid.

### 3. New copy

```json
{"text": "...", "intended_layer": "messaging", "channel": "landing_page"}
```

`intended_layer` doesn't restrict scoring — every layer is still scored — it only doubles that layer's weight in the overall verdict.

---

## Architecture

Embedding *generation* and embedding *use* are two different processes running
at two different times, in two different folders:

```
OFFLINE — vector_generation/generate_embeddings.py, run once per brand
────────────────────────────────────────────────────────────────────────
assets ──► Fingerprint (layered chunks) ──┐
                                           ├─► fit TfidfEmbedder (discriminative)
competitor corpus ──► GenericCorpus ──────┘        │
                                                    ▼
                          brand_vectors.npy, generic_vectors.npy (float32, L2-normalised)
                          brand_meta.json,  generic_meta.json     (text/layer/asset_type per row)
                          vectorizer.joblib, term_weights.npy     (the fitted encoder)
                          brand.index, generic.index              (FAISS IndexFlatIP)
                          ──► vector_generation/embeddings/<brand_id>/

RUNTIME — api.py / demo.py, no fitting or bulk-encoding ever happens here
────────────────────────────────────────────────────────────────────────
on startup: load every vector_generation/embeddings/<id>/ folder into memory
            (BrandVectorStore: vectors + FAISS indices + fitted encoder) and
            hold it there until the process stops.

new copy ──► InputEncoder.encode() ──► cosine vs BOTH centroids ──┐    per-layer
             (the ONLY live embedding call in the runtime path)   ├─►  C / D
             │                                                    │    ──► 2×2
             ├──► lexical  (MSTTR, signature hits, cliché density)┤
             ├──► stylometry (sentence len, FK grade, passive)    ┤
             ├──► tone (LLM-as-judge on 5 named axes)             ┤
             └──► FAISS nearest-neighbour lookup (evidence only) ─┘
```

| Layer | Consistency weights | Distinctiveness weights |
|---|---|---|
| Identity | centroid .60, signature .20, tone .20 | centroid .60, cliché .40 |
| Messaging | centroid .40, signature .20, structural .15, tone .25 | centroid .45, cliché .40, diversity .15 |
| Voice | centroid .20, signature .20, structural .25, tone .35 | centroid .30, cliché .45, diversity .25 |
| Positioning *(stretch)* | centroid .70, signature .30 | centroid .55, cliché .45 |

Voice is style-led, not semantics-led, on purpose: embeddings encode *topic* far more strongly than *tone*.

---

## Three implementation details that decide whether this works

**Calibration.** A raw cosine of 0.31 means nothing. Every score is rescaled between two anchors drawn from the corpora themselves — the brand's own leave-one-out self-similarity (what on-brand looks like *for this brand*) and the generic corpus (what boilerplate looks like). Chunks under 8 words are excluded from anchors; a one-word brand name has near-zero similarity to everything and drags every anchor to the floor.

**Discriminative reweighting.** Two language-learning companies both say "language", "learn", "lesson". Those terms carry no brand signal but eat most of the cosine. Terms are weighted by |log(brand rate / generic rate)| so distance reflects *how* a brand talks, not what industry it is in.

**Coverage discounting.** TF-IDF silently drops out-of-vocabulary words, so copy full of unfamiliar language collapses to whatever few words it shares with the brand — and scores as highly consistent on that remnant. Similarity is discounted by √coverage. (This correction is sparse-vector-specific; it disappears with dense embeddings.)

---

## Swapping in real embeddings

`vector_generation/embedder.py` (not `loci/`) is where the embedder lives now — that's the only module allowed to `.fit()` anything:

```python
from vector_generation.embedder import SentenceTransformerEmbedder
# then use it inside vector_generation/generate_embeddings.py's generate()
```

TF-IDF is the default because it needs no model download and runs anywhere — but cosine geometry is identical either way, so nothing downstream changes.

Tone uses an LLM judge when `ANTHROPIC_API_KEY` is set and falls back to a lexical heuristic otherwise, so the demo never fails live. The judge never sees the score it produces; it rates five named axes and the distance is computed in Python, which keeps the number reproducible and auditable.

---

## Generating embeddings

```
python -m vector_generation.generate_embeddings \
    --brand data/brand_duolingo.json \
    --generic data/generic_edtech.json \
    --out vector_generation/embeddings
```

Run this once per brand, and again whenever brand assets or the competitor corpus change. It writes `vector_generation/embeddings/<brand_id>/`:

| File | Contents |
|---|---|
| `brand_vectors.npy` / `generic_vectors.npy` | float32, L2-normalised vectors, one row per chunk |
| `brand_meta.json` / `generic_meta.json` | `{text, asset_type, layer, source_url, words}` per row, same order as the `.npy` — this is what makes it legible exactly what got embedded |
| `vectorizer.joblib` / `term_weights.npy` | the fitted, discriminatively-reweighted TF-IDF vectorizer — reused at runtime to embed new input copy, never refit |
| `brand.index` / `generic.index` | FAISS `IndexFlatIP` over the vectors above — the "vector base" the API mounts at startup |
| `manifest.json` | brand_id, brand_name, industry, dims, chunk counts, MVBF status |

`api.py` and `demo.py` never call this script's fitting code — they only load its output (`loci/vector_store.py`) and encode one new input at a time (`loci/input_encoder.py`). Restart the process to pick up freshly generated embeddings.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/brands` | list brands mounted at startup |
| POST | `/brands/{id}/score` | score new copy (fast, per request — the only live embedding call is encoding this one input) |
| GET | `/brands/{id}/vocabulary` | what the brand owns vs. what the category owns |

---

## Known limits

- Positioning is scaffolded but only meaningful with a real scraped competitor corpus.
- The generic centroid is only as good as the competitor list; a bad list produces a confidently wrong distinctiveness score.
- Thresholds (55/100) are tuned by hand on one brand, not learned.
- No scraper yet — assets are supplied as JSON.
- Adding a brand requires re-running the generation script and restarting the API process — there's no hot-reload of newly generated embeddings.
- The brand's tone profile (LLM judge over voice exemplars) is still computed at process startup, not precomputed by `vector_generation/` — it's a judgment call, not a vector, so it fell outside this refactor's scope.

*Demo asset text is written for this prototype in the style of the brand, not copied from live pages.*
