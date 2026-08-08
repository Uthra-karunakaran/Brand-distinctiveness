# Loci — Brand Distinctiveness Validation (prototype)

Answers two questions that most "brand voice" tools collapse into one:

- **Consistency** — *are you being you?* Measured against the brand's own fingerprint.
- **Distinctiveness** — *are you not sounding like everyone else?* Measured against a generic/competitor baseline.

They are computed from independent comparisons and combined only at the last step, into a 2×2.

```
python -m vector_generation.generate_embeddings   # offline: fit + embed once per brand
python demo.py                                    # worked example, four candidates, one per quadrant
uvicorn api:app --host 0.0.0.0 --port 8001                       # service — loads precomputed embeddings at startup
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

Competitor homepages, product pages, industry-report language, templated marketing copy. 15–30 chunks is enough to form a usable centroid. Owned by *industry*, not by brand — see "Industries are a separate, immutable resource" below — so most brands reference an already-registered industry instead of supplying this themselves.

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

`api.py` and `demo.py` never call this script's fitting code offline — the CLI wrapper (`generate()`) reads the two JSON files and calls the same `generate_from_dicts()` that the live `POST /brands/{id}/embeddings` endpoint calls with in-memory dicts instead of files. One fitting/encoding path, two callers. Restarting the process still works to pick up embeddings written by the CLI; it's just no longer required — see "Live embedding generation" below.

`generate()` also best-effort registers `--generic`'s `industry` into `vector_generation/industries/` if it isn't registered yet (a no-op if it already is) — see "Industries are a separate, immutable resource" below.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/brands` | list brands mounted in memory |
| POST | `/brands/{id}/score` | score new copy (fast, per request — the only live embedding call is encoding this one input) |
| GET | `/brands/{id}/vocabulary` | what the brand owns vs. what the category owns |
| POST | `/brands/{id}/embeddings` | submit assets + a reference to an industry corpus for background generation. Returns `202 {job_id, status: "queued"}` immediately; `409` if a job is already in flight for this `brand_id`. |
| GET | `/jobs/{job_id}` | poll a job's `status`, `stage`, `warnings`, `error`, and timestamps |
| GET | `/industries` | list every registered industry corpus, with `chunk_count` and which `brand_id`s currently reference it |
| GET | `/industries/{industry_id}` | one industry's stored corpus (`industry_id`, `items`, `chunk_count`) |

---

## Live embedding generation

`POST /brands/{id}/embeddings` is what onboarding calls instead of the CLI — same generation code (`generate_from_dicts()` in `vector_generation/generate_embeddings.py`), run in-process via FastAPI `BackgroundTasks`, with the result hot-loaded into `_SCORERS[brand_id]` on success. No restart, no queue, no database — job state lives in an in-memory dict (`vector_generation/jobs.py`) for the life of the process, the same pattern `_SCORERS` already uses.

### Industries are a separate, immutable resource

The generic/competitor corpus is owned by *industry*, not by *brand* — `vector_generation/industries/<industry_id>.json`, populated at startup the same way brand embeddings are (`vector_generation/industries.py`). An `industry_id`, once created, is **never edited or overwritten**. To change a corpus, register a new `industry_id` (e.g. `outdoor_gear_apparel_v2`) and re-point whichever brands should pick it up — brands still referencing the old id are left untouched, forever. There is deliberately no `PUT`/`POST /industries/{id}` update endpoint; the two `GET`s above are the entire industries surface.

`generic_corpus` is **mandatory** on every `POST /brands/{id}/embeddings` call, and `generic_corpus.industry` is always required. `generic_corpus.items` is optional, and only ever matters the *first* time an `industry_id` is seen:

- **`industry` already registered** → `items`, if sent, is ignored outright (`industry_corpus_ignored` warning) and the existing corpus is used. This is the common case — most brands reference an industry someone else (a brand or the CLI) already created.
- **`industry` not registered, `items` given** → this call creates that `industry_id` (one atomic write, persists regardless of whether the brand's own generation later succeeds or fails) and then proceeds using it.
- **`industry` not registered, no `items`** → hard fail, `unknown_industry`, before any fingerprinting starts.

Payload shape is identical either way — only whether `items` is present changes what happens:

```json
{
  "brand_name": "Linguacore",
  "assets": {"mission": "...", "tagline": "...", "blog": ["...", "..."]},
  "generic_corpus": {"industry": "language_learning_edtech"}
}
```

```json
{
  "brand_name": "TrailForge",
  "assets": {"mission": "...", "tagline": "...", "blog": ["...", "..."]},
  "generic_corpus": {
    "industry": "outdoor_gear_apparel",
    "items": [{"text": "...", "asset_type": "homepage"}]
  }
}
```

The CLI participates in the same registry: `generate()` best-effort registers `--generic`'s industry (if it isn't registered yet) as a side effect, so a brand generated offline is immediately reusable by a live brand afterward. Registration never changes what the CLI itself embeds — that always comes straight from the local `--generic` file, even if it has since diverged from what's registered.

**Job lifecycle** — `queued → running → ready | failed`, with `stage` set while `running`:

`fingerprinting → fitting_embedder → encoding_vectors → building_index → writing_manifest`

**Error codes** (job → `failed`, nothing written or hot-loaded — the existing brand, if any, keeps serving):

| Code | Cause |
|---|---|
| `unknown_industry` | `generic_corpus.industry` isn't registered and no `items` were sent to create it. |
| `mvbf_not_met` | `is_scorable()` is `False` — one of the 6 MVBF fields is missing, or there are fewer than 6 chunks total. `error.fields` lists the missing MVBF field names (empty for the chunk-count case). |
| `unmapped_asset_type` | An `asset_type` in `assets` isn't in `ASSET_LAYER_MAP`. |
| `internal_error` | Anything unexpected; message is the exception text. |

**Warning codes** (job still reaches `ready`):

| Code | Cause |
|---|---|
| `thin_generic_corpus` | The industry's corpus has fewer than 15 chunks — the centroid will be noisier than ideal. |
| `industry_corpus_ignored` | `generic_corpus.items` was sent for an `industry` that already exists — it was not used. |

**Rerun behaviour.** Calling `POST /brands/{id}/embeddings` again for the same `brand_id` overwrites `vector_generation/embeddings/<brand_id>/` (written to a temp dir and swapped in atomically, so a concurrent reader never sees a half-written folder) and hot-reloads the scorer — no versioning at the brand level, last write wins. The industry it references, however, is never rewritten by a rerun — see above. A second submission for a `brand_id` that already has a job in flight gets `409` rather than racing the first; industry creation has its own internal lock so two brands racing to create the same brand-new `industry_id` can't both win.

This is a single-process, in-memory job store and industry registry: neither survives a restart as *state* (though the industry JSON files and brand folders on disk do — a restart just re-reads them), and neither scales past one instance. That's fine for a prototype; if jobs need to coordinate across instances, swap `vector_generation/jobs.py` for a real queue (Redis/RQ/Celery) — the state machine and `/jobs/{id}` shape don't need to change to do that. The same applies to the delivery mechanism: the frontend polls `GET /jobs/{id}` today; an SSE stream could replace polling later without changing the job model.

---

## Known limits

- Positioning is scaffolded but only meaningful with a real scraped competitor corpus.
- The generic centroid is only as good as the competitor list; a bad list produces a confidently wrong distinctiveness score.
- Thresholds (55/100) are tuned by hand on one brand, not learned.
- No scraper yet — assets are supplied as JSON.
- The brand's tone profile (LLM judge over voice exemplars) is still computed at process startup, not precomputed by `vector_generation/` — it's a judgment call, not a vector, so it fell outside this refactor's scope.
- Job state (`vector_generation/jobs.py`) is in-memory only — it does not survive a process restart, and there's no cross-instance coordination. Fine for a single-process prototype; would need a real queue/store to go further.
- Industries are immutable by design — there is no way to fix a typo or bad entry in an already-registered industry's corpus in place. The only path is registering a new `industry_id` and re-running the brands that should use it.

*Demo asset text is written for this prototype in the style of the brand, not copied from live pages.*
