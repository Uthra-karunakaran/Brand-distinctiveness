# The Loci Engine — A Complete Field Guide

This is not a quick-reference README (that's `README.md`, one folder over). This is a
line-by-line, concept-by-concept walkthrough of every piece of this backend — what
problem it solves, what math or heuristic it uses, why that math and not something
simpler, and exactly how it wires into the API. The goal is that by the end you could
re-derive any number this service returns, by hand, from the input text.

Read it in order the first time. After that, use the Table of Contents as a reference.

---

## Table of Contents

- **Part 0** — Why this exists
- **Part 1** — Core concepts, from scratch (read this before anything else)
- **Part 2** — The data model (`loci/fingerprint.py`)
- **Part 3** — Offline pipeline: assets → vectors on disk (`vector_generation/`)
- **Part 4** — Runtime loading (`loci/vector_store.py`, `loci/input_encoder.py`)
- **Part 5** — The scoring engine, metric by metric (`loci/metrics/*`, `loci/scorer.py`)
- **Part 6** — The API, endpoint by endpoint (`api.py`)
- **Part 7** — Cross-cutting concerns (rate limits, LLM economics, deployment)
- **Part 8** — A complete worked trace, with real numbers from this exact repo
- **Appendix A** — Glossary
- **Appendix B** — File map

---

# Part 0 — Why This Exists

Most "brand voice checker" tools collapse two very different questions into one score:

1. **Consistency** — does this new piece of copy sound like *this brand*, specifically?
2. **Distinctiveness** — does this copy sound different from *everyone else in the
   category*, or could it have been written by any competitor?

Collapsing these into one number hides two opposite failure modes:

- Copy that is perfectly on-brand but is also exactly what every competitor says
  ("empowering learners worldwide with innovative solutions") — **consistent, not
  distinctive.**
- Copy that stands out but doesn't sound like the brand at all — **distinctive, not
  consistent.**

Loci keeps these as two separate numbers all the way through the pipeline, and only
combines them at the very last step, into a 2×2 grid (a "quadrant"), never into a
blended average. That single design decision — *never blend, always report both
axes* — is the reason nearly every module in this codebase computes things in pairs
(a brand-facing measurement and a generic/competitor-facing measurement), from
independent data.

Everything else in this document is the mechanics of computing those two numbers
honestly.

---

# Part 1 — Core Concepts, From Scratch

Skip nothing in this part even if some of it looks familiar — later chapters assume
you have internalized the exact formulas here, not just the general idea.

## 1.1 Vectors and "meaning as geometry"

The engine's core trick, used everywhere, is: turn a piece of text into a list of
numbers (a **vector**), such that texts with similar meaning/vocabulary end up as
vectors that point in similar *directions*. Once text is a vector, "how similar are
these two pieces of text" becomes "how similar are these two directions" — a
geometry problem, which is cheap and well-understood, instead of a language problem,
which isn't.

This repo uses two different ways to build such vectors, at two different layers of
the stack:

- **TF-IDF vectors** (`vector_generation/embedder.py`) — sparse, built from word/phrase
  counts. This is what's actually running in this deployment.
- **Sentence-embedding vectors** (`SentenceTransformerEmbedder`, same file) — dense,
  built by a neural network. Wired in as a drop-in alternative, not currently used
  (no model is downloaded in this deployment).

Both produce a vector per text; everything downstream (centroids, cosine similarity,
FAISS) only cares that the vector exists and is comparable to others in the same
space — it does not care how the vector was built. That's a deliberate seam: swapping
the embedder never requires touching the scoring logic.

## 1.2 TF-IDF, term by term

TF-IDF ("term frequency, inverse document frequency") turns a document into a vector
of one number per vocabulary word (or word-pair), where the number answers: *"how much
does this term define this document, relative to how common it is everywhere?"*

- **TF (term frequency)** — how often the term appears in this specific document. A
  raw count would let a long document dominate just by repeating a word; this repo
  configures `sublinear_tf=True` in `TfidfVectorizer`, which uses `1 + log(count)`
  instead of the raw count, so going from 1 occurrence to 2 matters a lot, but going
  from 20 to 21 barely does.
- **IDF (inverse document frequency)** — how *rare* the term is across the whole
  corpus the vectorizer was fit on. A term that appears in every document (like "the",
  or in this domain, "language") gets a low IDF weight, because it doesn't distinguish
  one document from another. A term that appears in only a couple of documents gets a
  high IDF weight.
- **TF-IDF = TF × IDF.** High when a term is used a lot *in this document* and rarely
  elsewhere. Low otherwise.

The vectorizer is configured in `vector_generation/embedder.py:38-44`:

```python
self._vec = TfidfVectorizer(
    sublinear_tf=True,
    ngram_range=(1, 2),
    min_df=1,
    stop_words=None,       # stopwords ARE voice signal — keep them
    lowercase=True,
)
```

Three choices worth understanding, because they're not the sklearn defaults:

- `ngram_range=(1, 2)` — the vocabulary includes single words *and* two-word phrases
  ("free trial", "streak alive"). Phrases carry brand voice that single words miss.
- `min_df=1` — a term only needs to appear once anywhere in the fitting corpus to earn
  a vocabulary slot. With a corpus this small (tens of chunks, not millions of
  documents), requiring a higher minimum document frequency would throw away most of
  the vocabulary.
- `stop_words=None` — sklearn's default English stopword list ("the", "is", "we", "you"...)
  is deliberately **not** removed. Most TF-IDF setups strip stopwords because they carry
  no *topic* signal. But this system also uses TF-IDF-adjacent word counts as a stand-in
  for *voice*, and function words are exactly what distinguishes a brand that writes
  "we" and "our" (corporate) from one that writes "you" and "your" (direct-address) —
  see the tone heuristic in §5.4. Stripping them would delete the signal this system
  needs.

The result of fitting is a **vocabulary** (a fixed list of terms this brand+industry's
text ever used) and, for each document, a sparse vector: mostly zeros, with a nonzero
entry for every vocabulary term the document actually contains.

**Fitting vs. encoding** — this distinction matters throughout the codebase:

- **Fit** = build the vocabulary and IDF weights from a corpus. Happens exactly once
  per brand, offline, in `vector_generation/embedder.py`. Nothing at request time is
  ever allowed to fit anything (`loci/input_encoder.py`'s docstring calls this out
  explicitly — it "has no `.fit()`").
- **Encode / transform** = project a new piece of text through an *already-fitted*
  vectorizer into that same vector space. This is cheap (just counting which known
  vocabulary terms are present) and is the only thing allowed to happen at request
  time.

## 1.3 Cosine similarity, and why everything gets L2-normalized

Once two texts are vectors, "how similar are they" is measured as the **cosine of the
angle between them** — 1.0 means pointing in exactly the same direction (same
relative word usage), 0 means unrelated (orthogonal), and it's insensitive to vector
*length*, which matters because a long document and a short one that use the same
words in the same proportions should be considered similar even though the long one's
raw word counts are bigger.

Mathematically, cosine similarity is:

```
cosine(a, b) = (a · b) / (|a| · |b|)
```

If both `a` and `b` are already scaled to length 1 (**L2-normalized** — divided by
their own Euclidean norm), the denominator is just `1 × 1 = 1`, so cosine similarity
becomes a plain dot product. That's exactly what `loci/vectormath.py` exploits:

```python
def l2_normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both already L2-normalised
```

Every vector that exists anywhere in this system — brand vectors, generic vectors,
centroids, a freshly encoded input — is L2-normalized at the point it's created
(`TfidfEmbedder.encode()`, `InputEncoder.encode()`, `centroid()` all end with
`l2_normalise(...)`). That's what lets `cosine()` skip the division entirely and lets
FAISS use a plain inner-product index (§1.9) instead of a slower true-cosine index.

The `norms[norms == 0] = 1.0` line guards against dividing by zero for an all-zero
vector (e.g. a piece of input text with zero vocabulary overlap) — it leaves that
vector as all zeros rather than crashing on `0/0`.

## 1.4 Centroids: averaging in vector space

A **centroid** is the "average direction" of a set of vectors — literally the mean,
re-normalized back onto the unit sphere so it's comparable to the individual vectors
it was built from:

```python
def centroid(vectors: np.ndarray) -> np.ndarray:
    """Mean-pool then re-normalise, so centroid lives on the same unit sphere."""
    c = vectors.mean(axis=0, keepdims=True)
    return l2_normalise(c)[0]
```

This system computes exactly two centroids per brand:

- **Brand centroid** — the mean direction of every chunk of the brand's own language
  (tagline, mission, blog posts, ads, everything).
- **Generic centroid** — the mean direction of the competitor/industry corpus.

Both are computed once (per layer, see §5.1) when the scorer is built, and every
scoring request compares the new input's vector against both of these fixed points —
never against individual chunks directly (except for the FAISS "nearest evidence"
lookup, which is separate and explained in §1.9).

## 1.5 Calibration: why a raw cosine number is meaningless

A cosine similarity of 0.31 between a new piece of copy and the brand centroid tells
you almost nothing on its own. Is that good? It depends entirely on:

- how big the vocabulary is (more dimensions → generally lower absolute cosines),
- how tight or diffuse the brand's own corpus is,
- how similar the industry's generic language already is to the brand's.

Comparing raw cosines across brands, or even trusting a single raw number for one
brand, is a trap. The fix used throughout `loci/metrics/centroid.py` is **calibration
against anchors drawn from the corpora themselves** — i.e., instead of asking "is 0.31
big?", ask "where does 0.31 fall between the two experimentally-observed cosines that
represent *this brand's own definition* of 'typical on-brand' and 'typical generic'?"

Four anchors are computed, two per axis:

| Axis | "Low" anchor (→ score 0) | "High" anchor (→ score 100) |
|---|---|---|
| Consistency | generic chunks' similarity **to the brand centroid** (60th percentile) | brand chunks' own leave-one-out similarity **to their own centroid** (85th percentile) |
| Distinctiveness | generic chunks' leave-one-out similarity **to their own centroid** (40th percentile) | brand chunks' similarity **to the generic centroid** (60th percentile) |

Read the distinctiveness row carefully — it's the one that trips people up. High
*distinctiveness* means the input does **not** look like generic/competitor language.
So the anchor that maps to a distinctiveness score of 100 is "how far a typical brand
chunk sits from the generic centroid" (i.e. brand-vs-generic similarity — usually
*low*), and the anchor that maps to a distinctiveness score of 0 is "how tightly
generic chunks cluster around their own centroid" (generic-vs-generic similarity —
usually *high*). The anchors are numerically inverted on the raw-cosine number line,
and the rescaling function is built to handle that (see below).

The actual rescale function, `loci/metrics/centroid.py:35-40`:

```python
def _rescale(value: float, low: float, high: float) -> float:
    """Map value from [low, high] onto [0, 100], clamped. Handles inverted ranges."""
    if abs(high - low) < 1e-9:
        return 50.0
    pct = (value - low) / (high - low)
    return float(max(0.0, min(1.0, pct)) * 100.0)
```

This is min-max scaling: `pct` is how far `value` sits along the line from `low` to
`high`, clamped to `[0, 1]` so a value outside the observed anchor range doesn't
produce a score outside `[0, 100]`. Because the formula is a plain ratio (not an
`if low < high` branch), it works identically whether `low < high` or `low > high` —
which is exactly the inverted case distinctiveness needs. Worked example: suppose
`dist_generic_anchor` (low) = 0.50 and `dist_brand_anchor` (high) = 0.05 (brand chunks
really don't look like the generic centroid). An input with raw similarity-to-generic
of 0.05 gets `pct = (0.05 - 0.50) / (0.05 - 0.50) = 1.0` → **100** (maximally
distinctive). An input with raw similarity of 0.50 (looks exactly like the generic
centroid) gets `pct = 0` → **0**.

If `high == low` (degenerate — e.g. every calibration chunk scored identically),
`_rescale` returns a neutral 50.0 rather than dividing by (near) zero.

## 1.6 Leave-one-out validation

If you want to know "how similar is a typical brand chunk to the brand centroid," the
naive approach — compute the centroid from *all* chunks, then measure each chunk
against it — is biased: every chunk contributed to the centroid it's being measured
against, so it's partly measuring a chunk against itself. This inflates the anchor,
especially with small corpora where one chunk can visibly shift the centroid.

**Leave-one-out (LOO)** fixes this: for each chunk, recompute the centroid *excluding
that chunk*, then measure the chunk against that centroid instead.

```python
def _loo_sims(self, vecs: np.ndarray) -> list[float]:
    sims = []
    for i in range(len(vecs)):
        rest = np.delete(vecs, i, axis=0)
        sims.append(cosine(vecs[i], centroid(rest)))
    return sims
```

This is O(n²)-ish (recomputes a centroid for every chunk), which is fine at the scale
of tens of chunks and would need to change if a brand's corpus ever grew into the
thousands. It's used for both of the "self-similarity" anchors: `brand_self` (feeds
`cons_high`) and `generic_self` (feeds `dist_generic_anchor`) in
`centroid.py:_calibrate()`.

## 1.7 Discriminative reweighting (the "lift" trick)

Take two language-learning competitors. Both write "learn a language," "lesson,"
"fluency." Raw TF-IDF gives high weight to whatever's rare across the *whole* fitting
corpus — but if "language" and "lesson" are common in *both* the brand's own text and
the competitor corpus, a plain TF-IDF cosine between two language-learning brands will
be dominated by these shared, topic-defining-but-brand-neutral terms. The result: two
completely different-sounding language apps would still look "similar" to each other,
because the similarity is really measuring "both are about language learning," not
"both sound the same."

The fix, in `TfidfEmbedder.fit_discriminative()` (`vector_generation/embedder.py:65-90`):

1. Fit the vectorizer on the **joint** corpus (brand chunks + generic chunks
   together), so both share one vocabulary and one vector space.
2. For every vocabulary term, compute its relative rate in the brand corpus (`b`) and
   in the generic corpus (`g`) — each a probability distribution over the vocabulary
   (counts divided by the corpus's total token count).
3. Compute `lift = |log((b + ε) / (g + ε))|` for every term — this is large when a term
   is used at very different rates in the two corpora (strongly brand-specific *or*
   strongly generic-specific — the absolute value means both directions count), and
   near zero when a term is used at similar rates in both (shared, non-discriminative
   vocabulary like "language" or "learn").
4. Normalize by the max lift observed, so weights land in `[0, 1]`, then clip the floor
   at `0.05` (so no term is ever fully zeroed out) — giving a `term_weights` array,
   one entry per vocabulary term.
5. Every future encode of any text (brand chunk, generic chunk, or a live input at
   request time) multiplies its raw TF-IDF vector element-wise by `term_weights`
   *before* L2-normalizing.

The practical effect: shared category vocabulary gets dialed down, so cosine
similarity between two vectors increasingly reflects the *brand-specific* word choices
they share, not the fact that they're both about the same general topic. The small
`strength` exponent parameter (`(lift/max_lift) ** strength`, default `1.0`) exists to
let a future caller sharpen or soften this reweighting; it's not currently varied
anywhere in the codebase.

`term_weights.npy` is persisted alongside the vectorizer (`vectorizer.joblib`) so that
`loci/input_encoder.py` — which has no fitting logic at all — can apply the exact same
weighting to a brand-new piece of input text at request time.

## 1.8 Coverage discounting (the out-of-vocabulary problem)

TF-IDF has a blind spot: any word that wasn't in the fitting vocabulary is **silently
dropped** — it contributes nothing to the vector, positive or negative. This creates a
specific failure mode. Imagine a piece of copy written almost entirely in vocabulary
the brand and its competitors never use ("enrolment," "tuition," "supervised
sessions," for a brand that's never once used those words). TF-IDF reduces that input
to whatever handful of ordinary words it happens to share with the brand ("the",
"and," maybe "learn") — and if those few shared words happen to cosine well against
the brand centroid, the system would report **high consistency**, on the strength of
noise, for a piece of copy that's actually saying something the brand has never said.

The fix is **coverage discounting** — measure what fraction of the input's tokens
actually exist in the fitted vocabulary, and discount the similarity score by that
fraction (via a square root, so the penalty is gentler than linear — a text that's
70% in-vocabulary shouldn't lose 30% of its score, just some of it):

```python
def coverage(self, text: str) -> float:
    toks = self._vec.build_analyzer()(text)
    if not toks:
        return 0.0
    seen = sum(1 for t in toks if t in self._vec.vocabulary_)
    return seen / len(toks)
```

`build_analyzer()` returns sklearn's actual tokenizer/n-gram function — so `toks`
includes both unigrams and bigrams, matching exactly what the vectorizer itself would
have extracted. `coverage()` is duplicated in two places on purpose: once in
`TfidfEmbedder` (`vector_generation/embedder.py`, used if you ever score something
during generation) and once in `InputEncoder` (`loci/input_encoder.py`, used at
request time) — both read from the same persisted `vectorizer.joblib`, so they always
agree.

The discount is applied in `DualCentroidModel.score()` (`loci/metrics/centroid.py:106-122`):

```python
sb = cosine(v, self.brand_centroid)
sg = cosine(v, self.generic_centroid)

cov = getattr(self.encoder, "coverage", None)
if cov is not None:
    sb *= cov(text) ** 0.5
```

**Note the asymmetry** — this discount is only applied to `sb` (similarity to the
*brand* centroid, which feeds the consistency score), not to `sg` (similarity to the
generic centroid, which feeds distinctiveness). A piece of copy in totally unfamiliar
vocabulary will therefore have its consistency score pulled down by low coverage, but
its raw distinctiveness measurement is untouched by this particular correction. That's
a real, current asymmetry in the code, not a typo you should assume is a bug — but it
is worth knowing if you ever find a distinctiveness score that looks unexpectedly high
for very off-vocabulary input.

This whole correction is **specific to sparse, vocabulary-based embeddings**. A dense
sentence-transformer model (§1.1's second embedder option) has no concept of
"out-of-vocabulary" — every word maps to *something* in a dense space — so if the
embedder is ever swapped, `coverage()` naturally becomes a no-op (`getattr(...,
"coverage", None)` returns `None` for an encoder that doesn't define it, and the
discount is skipped entirely).

## 1.9 FAISS and nearest-neighbor search

FAISS (Facebook AI Similarity Search) is a library for fast "find the k most similar
vectors to this one" queries over a fixed set of vectors. At the scale of tens or
hundreds of chunks per brand, a brute-force loop would be plenty fast — FAISS is used
here more for the *interface* (a persistent, on-disk index) and future headroom than
because brute force wouldn't work today.

Two indices are built per brand, at generation time (`vector_generation/generate_embeddings.py:94-97`):

```python
def _faiss_index(vecs: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vecs.shape[1])  # vectors are L2-normalised -> inner product == cosine
    index.add(vecs)
    return index
```

`IndexFlatIP` = a "flat" (exhaustive, exact — no approximation) index that ranks
matches by **inner product**. Because every vector fed into it is already
L2-normalized (§1.3), inner product *is* cosine similarity here — no extra
normalization step needed inside FAISS itself. "Flat" means it's not using any
approximate-nearest-neighbor trick (like clustering or graph search) — it checks every
vector, guaranteeing exact results, which is appropriate at this corpus size and
avoids introducing approximation error into a demo whose whole point is
interpretability.

Two indices are built: one over brand chunks (`brand.index`), one over generic chunks
(`generic.index`) — both loaded once at startup by `BrandVectorStore`
(`loci/vector_store.py:39-40`) and queried once per scoring request purely for
**evidence**, not for the score itself:

```python
def nearest_brand_chunks(self, vec: np.ndarray, k: int = 3) -> list[str]:
    _, i = self.brand_index.search(
        vec.reshape(1, -1).astype("float32"), min(k, self.brand_index.ntotal))
    return [self.brand_meta[int(idx)]["text"] for idx in i[0] if idx != -1]
```

These are the `nearest_brand_chunks` / `nearest_generic_chunks` lists you see in every
`/score` response's `evidence` block — literally "here are the 3 real chunks (from the
brand's own corpus, and from the generic corpus) that this input's vector landed
closest to," so a human reviewing the score has something concrete to look at, not
just a number.

## 1.10 Lexical statistics: vocabulary richness, signatures, and clichés

`loci/metrics/lexical.py` computes three signals with zero machine learning — plain
counting.

**Type-token ratio (TTR)** is the classic vocabulary-richness metric: unique words
divided by total words. Its well-known flaw is that it's length-biased — a longer text
naturally reuses words more, so TTR drops as length increases, which makes it useless
for comparing texts of different lengths. The fix used here is **MSTTR (Mean
Segmental TTR)**: chop the token stream into fixed-size windows (40 tokens), compute
TTR *within each window*, and average the per-window ratios. Because every window is
the same length, the length bias disappears.

```python
def msttr(tokens: list[str], window: int = 40) -> float:
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = [
        len(set(tokens[i:i + window])) / window
        for i in range(0, len(tokens) - window + 1, window)
    ]
    return sum(ratios) / len(ratios)
```

**Signature terms and cliché terms** use the same "lift" idea as §1.7, but on plain
word-count rates instead of vector weights, and they're computed once per brand
(cached as sets on `LexicalScorer`) rather than per request:

```python
def signature_terms(brand, generic, top_k=40, min_count=2, min_lift=2.0):
    scored = []
    for term, n in brand.counts.items():
        if n < min_count or len(term) < 4 or term in STOP:
            continue
        lift = (brand.rate(term) + 1e-6) / (generic.rate(term) + 1e-6)
        if lift >= min_lift:
            scored.append((term, lift))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]
```

A **signature term** is a word the brand uses at least `min_lift` (2×) more often,
relative to its own vocabulary, than the generic/competitor corpus does — filtered to
words of at least 4 letters (drops short function words) and excluding a hand-written
stopword list (`STOP`, `lexical.py:22-28` — a second, smaller stopword list than
sklearn's, specifically tuned to keep out words that are common everywhere but not
meaningfully "brand voice," like "because," "should," "would"). A **cliché term** is
the mirror image: a word the *generic* corpus over-uses relative to *this brand*. The
`min_lift` floor is deliberate: it's what stops a word that both sides use at similar
rates (like "free" for a brand for which it's core, and "free" for a whole industry
that also says it constantly) from being claimed as either a signature or a cliché —
it just sits in neither list.

`LexicalScorer.score(text)` then checks a new piece of input against both
pre-computed sets:

```python
content = [t for t in toks if len(t) >= 4]
hit = sorted({t for t in content if t in self.signatures})
clich = sorted({t for t in content if t in self.cliches})

denom = max(len(set(content)), 1)
sig_rate = len(hit) / denom
cli_rate = len(clich) / denom

return LexicalScores(
    diversity=round(min(msttr(toks) / 0.75, 1.0) * 100, 1),
    signature_hit=round(min(sig_rate / 0.25, 1.0) * 100, 1),
    cliche_free=round(max(1.0 - cli_rate / 0.30, 0.0) * 100, 1),
    ...
)
```

Both rates are normalized against a hand-picked "saturation point," not against 100%:
`signature_hit` treats **25% signature-term density** among the input's unique content
words as already maximal (score caps at 100 once a quarter of your distinct
substantive words are brand signatures — real copy is never going to be *all*
signature terms, so requiring more would make the score unreachable). `cliche_free`
treats **30% cliché density** as the point where the score bottoms out at 0 (fully
generic copy). These two constants (`0.25`, `0.30`) are hand-tuned thresholds, not
learned — see the "Known limits" note in Part 7.

`signature_hit` feeds **consistency** (using signature terms = sounding like the
brand); `cliche_free` feeds **distinctiveness** (avoiding cliché terms = not sounding
like everyone else) — the same brand-vs-generic split every other metric in this
system follows.

## 1.11 Stylometry: sentence shape, readability, and style envelopes

`loci/metrics/structural.py` treats "how a brand writes" (independent of *what* words
it uses) as a small feature vector per text — six numbers:

| Feature | What it measures |
|---|---|
| `mean_sentence_len` | average words per sentence |
| `sentence_len_sd` | how much sentence length *varies* (a brand that alternates short punchy sentences with longer ones has high SD; one that writes uniformly does not) |
| `flesch_kincaid` | US school-grade reading level (see formula below) |
| `bullet_ratio` | fraction of lines that are bullet points or numbered list items |
| `passive_ratio` | passive-voice constructions per sentence (regex heuristic, not a real parser) |
| `exclaim_per_100w` | exclamation marks per 100 words |

The **Flesch-Kincaid Grade Level** formula (a decades-old, well-known readability
formula, used as-is):

```
grade = 0.39 × (words / sentences) + 11.8 × (syllables / words) − 15.59
```

implemented directly in `structural.py:55`. Syllable counting is a crude heuristic —
count consecutive vowel-letter groups in a word after stripping a trailing "e"
(`syllables()`, `structural.py:24-26`) — not a linguistically correct syllabifier, but
close enough for a relative comparison against the brand's own baseline (it doesn't
need to be *accurate*, just *consistent*, since it's always compared against the same
method run on the brand's own corpus).

`PASSIVE = re.compile(r"\b(?:is|are|was|were|be|been|being)\b\s+\w+(?:ed|en)\b", re.I)`
is a deliberately simple regex, not a grammatical parser — it will have false
positives/negatives on edge cases, but again, it only needs to be a stable, repeatable
measurement, applied identically to the brand's baseline and to new input.

**The style envelope**: `StructuralScorer.__init__` computes this 6-feature vector for
every brand text, then takes the mean and standard deviation of each feature *across
the brand's whole corpus* — this is the brand's "style envelope." One detail worth
noting: the standard deviation is floored —

```python
self.sd = [max(st.pstdev(c), abs(st.mean(c)) * 0.25, 0.5) for c in cols]
```

— at whichever is largest of: the real observed SD, 25% of the mean, or `0.5`. Without
this floor, a brand whose corpus happens to be extremely uniform on some feature (say,
every single chunk has almost exactly the same sentence length) would produce a tiny
SD, which would make the *z-score* (next paragraph) blow up for any new input that
deviates even slightly — an artificially "infinitely tight" envelope that would flag
almost anything as off-style. Flooring the SD keeps the envelope realistically
forgiving.

Scoring a new text is a **z-score** against that envelope — how many standard
deviations away from the brand's mean is this feature, averaged across all six
features:

```python
z = abs(val - mu) / sd
...
mean_z = sum(zs) / len(zs)
style_match = round(max(0.0, 1 - mean_z / 2.5) * 100, 1)
```

A `mean_z` of 0 (dead-on average across every feature) scores 100. A `mean_z` of 2.5 or
more (two-and-a-half standard deviations off, on average, across six stylistic
dimensions) scores 0. `2.5` is, again, a hand-picked constant, not learned from data.

## 1.12 LLM-as-judge, and why tone specifically needs one

Every other metric in this system is either pure vector math or pure counting — no
network calls, no model inference beyond a fitted TF-IDF transform. **Tone is the one
exception**, and the reason is explained directly in `loci/metrics/tone.py`'s
docstring: embeddings (and by extension, TF-IDF cosine) encode *topic* far more
strongly than *tone*. "We help teams ship faster" and "We empower organisations to
accelerate delivery" sit close together in TF-IDF space (they share most of their
substantive vocabulary) despite being tonally opposite — one casual and direct, one
corporate and inflated. No amount of reweighting fixes this, because the two sentences
really *are* about the same topic; the difference is entirely in register, which
vector-space-of-words methods aren't built to capture.

So tone gets a different kind of judge: an LLM, prompted to rate a sample on five
named axes, 0–10 each:

```python
AXES = ["formal_casual", "serious_playful", "corporate_human",
        "restrained_bold", "technical_accessible"]

JUDGE_SYSTEM = """You are a brand-voice rating instrument. You do not write copy \
and you do not give opinions.

Rate the SAMPLE on each axis from 0 to 10:
- formal_casual: 0 = highly formal, 10 = highly casual
...
Return ONLY a JSON object mapping each axis name to an integer. No prose, no \
markdown fences."""
```

**The critical design choice**: the judge is never asked "is this on-brand?" or given
the final score to produce. It only ever rates raw text on these five fixed axes — the
same prompt, whether it's rating one of the brand's own voice exemplars or a brand-new
input. The *distance* between two axis-profiles (brand's own profile vs. the input's
profile) is then computed in plain Python (mean absolute gap across the five axes,
§5.4). This keeps the actual 0–100 number **reproducible and auditable** — it never
depends on the LLM's opinion of a specific score, only on its (much narrower, much
more consistent) ability to rate "how casual is this text from 0 to 10," which is a
far easier and more stable task to get consistent answers to than "does this sound
like brand X."

If no `ANTHROPIC_API_KEY` is configured (true of the current Render deployment — see
Part 7), or the call errors for any reason, the system falls back to a **lexical
heuristic** that approximates the same five axes using pure regex/word-list signals
(contraction rate, a hand-written corporate-jargon word list, exclamation rate,
second-person vs. first-person-plural pronoun rates, long-word rate, average sentence
length). It's a cruder proxy — it can't detect genuine wit or true formality the way an
LLM reading for meaning can — but it means the service never hard-fails just because a
key is missing or an API call times out.

## 1.13 The quadrant model

Once consistency and distinctiveness are both on a 0–100 scale, the system draws one
line on each axis — a threshold, currently `55.0`, hand-tuned rather than learned
(`loci/scorer.py:60`) — and reports which of four quadrants the input landed in:

| | Distinctive (≥55) | Generic (<55) |
|---|---|---|
| **Consistent (≥55)** | **IDEAL** — on-brand and stands out | **ON-BRAND BUT GENERIC** — sounds like the brand, but so does everyone in the category |
| **Inconsistent (<55)** | **UNIQUE BUT OFF-BRAND** — stands out, but doesn't sound like this brand | **LOST** — neither |

This 2×2 (`QUADRANTS` dict, `loci/scorer.py:51-58`) is computed independently at
**every layer** (identity/messaging/voice/positioning — see Part 2) and once more for
the overall verdict — it is never averaged away into a single number, by design (Part
0).

---

# Part 2 — The Data Model (`loci/fingerprint.py`)

Before anything gets embedded or scored, raw brand material has to be normalized into
a common shape. That shape is a `Chunk`: one piece of embeddable brand language,
tagged with which **layer** it belongs to.

## 2.1 Layers and the asset-to-layer map

```python
class Layer(str, Enum):
    IDENTITY = "identity"       # Who are we?
    MESSAGING = "messaging"     # What do we repeatedly communicate?
    VOICE = "voice"             # How do we sound?
    POSITIONING = "positioning"  # Why choose us?  (stretch layer)
    PROOF = "proof"             # Can we back it up?
```

Scoring is reported *per layer*, not as one global number, because "is this input
on-brand" means something different depending on what kind of copy it is — a job
posting and a tagline shouldn't be judged by the same yardstick. Every raw asset type
a brand might supply gets mapped to exactly one layer via a fixed lookup table,
`ASSET_LAYER_MAP` (`fingerprint.py:29-53`):

| Asset type | Layer | Asset type | Layer |
|---|---|---|---|
| `name`, `tagline`, `mission`, `vision`, `values`, `about`, `founder_story` | Identity | `homepage`, `product_page`, `landing_page`, `cta`, `email` | Messaging |
| `blog`, `social`, `ad`, `support_doc`, `job_post` | Voice | `sales_deck`, `investor_deck`, `comparison_page` | Positioning |
| `case_study`, `testimonial`, `review` | Proof | | |

Note that `PROOF` is a defined layer with asset types mapped to it, but it does **not**
appear in `LAYER_WEIGHTS` (§5.5) — it's collected into the fingerprint (so
`case_study`/`testimonial`/`review` assets are accepted and not rejected) but never
actually scored. It's scaffolding for a future layer, not a bug.

Any asset type *not* in this table is rejected loudly at ingest time — `from_assets()`
raises `ValueError(f"Unmapped asset_type '{asset_type}'...")` rather than silently
dropping or mis-filing it (`fingerprint.py:96-99`). This is what produces the
`unmapped_asset_type` job error code you'll see in Part 6.

## 2.2 Chunk, BrandFingerprint, and the MVBF floor

A `Chunk` is the atomic unit: `{text, asset_type, layer}`. A
`BrandFingerprint` is just `{brand_id, brand_name, chunks: [Chunk]}`, built from a raw
`{asset_type: text | [texts]}` dict via `BrandFingerprint.from_assets()`.

**MVBF** = "Minimum Viable Brand Fingerprint" — six required asset types:

```python
MVBF_FIELDS = ("name", "tagline", "mission", "vision", "values", "about")
```

`is_scorable()` is the gate every brand must pass before it can be scored at all:

```python
def is_scorable(self) -> tuple[bool, str]:
    status = self.mvbf_status()
    missing = [f for f, ok in status.items() if not ok]
    if missing:
        return False, f"Missing MVBF fields: {', '.join(missing)}"
    if len(self.chunks) < 6:
        return False, "Fewer than 6 chunks — centroid would be unstable."
    return True, "ok"
```

Two independent floors: all six MVBF asset types must be present (as at least one
chunk each), **and** the fingerprint must have at least 6 total chunks (in practice
this second check is almost always already satisfied once the MVBF is met, since MVBF
alone supplies 6 asset types — but if any MVBF field's text was empty/whitespace and
got dropped by `from_assets()`'s `if t:` filter, the chunk count could fall below 6
even with all field *names* nominally present as keys in the input dict). Below this
floor, the brand centroid itself would be built from too little material to mean
anything — this isn't a data-quality nicety, it's a "there is not enough information
to compute a stable centroid" hard stop.

## 2.3 GenericCorpus — the second reference point

Structurally identical to a fingerprint but built from a flat list of
`{text, asset_type}` dicts rather than the asset-dict shape (competitor
corpora don't need MVBF-style structure — they're just "language from this
industry"). `GenericCorpus.texts(layer)` has a fallback baked in: if fewer than 3
chunks exist for the requested layer, it returns the *entire* corpus instead —
competitor corpora are typically thin per-layer, and 1-2 chunks isn't enough to build
anything meaningful, so falling back to "all of it" is better than a near-empty
comparison set. `BrandVectorStore.generic_layer_vectors()` (§4.1) mirrors this same
`< 3 → fall back to everything` rule at the vector level.

## 2.4 InputCopy — the thing actually being scored

```python
class InputCopy(BaseModel):
    text: str
    intended_layer: Layer = Layer.MESSAGING
    channel: str = "landing_page"
    label: str | None = None
```

`intended_layer` does **not** restrict which layers get scored — every present layer
is always scored — it only doubles that layer's weight when the *overall* verdict is
aggregated across layers (see `agg()` in §5.5). `channel` is accepted but not currently
read anywhere in the scoring logic — it exists in the schema as a hook for a future
per-channel adjustment, not something that changes today's output.

---

# Part 3 — Offline Pipeline: Assets → Vectors on Disk

Everything in this part happens **once per brand**, never at request time (with one
exception covered in Part 6: the live `POST /brands/{id}/embeddings` endpoint runs
this exact same code, just triggered by an API call instead of a CLI invocation, in a
background thread).

## 3.1 `vector_generation/embedder.py` — already covered in depth in §1.2, §1.7, §1.8.
One thing worth restating structurally: this module is **the only place `.fit()` is
ever called** anywhere in the codebase. `loci/` (the runtime package) has no import
path to anything with a `.fit()` method — `loci/input_encoder.py`'s `InputEncoder`
only has `.load()` and `.encode()`. This is an architectural invariant, not just a
convention: it's *impossible*, by the shape of the code, for the live API to
accidentally re-fit a vector space in response to a request.

## 3.2 `generate_from_dicts()` — the shared core, stage by stage

Both the CLI (`generate()`) and the live API endpoint
(`api.py`'s `_run_embedding_job`) funnel through one function,
`generate_from_dicts()` (`vector_generation/generate_embeddings.py:100-213`). Here's
what happens, in order, each stage reported via an optional `on_stage` callback (this
is what powers the live job's `stage` field, Part 6):

1. **`FINGERPRINTING`** — `BrandFingerprint.from_assets(...)` and
   `GenericCorpus.from_texts(...)` turn the raw dicts into layered `Chunk` lists (Part
   2). `fp.is_scorable()` is checked immediately; if `strict=True` (the live-endpoint
   path) and it fails, a `ScorabilityError` is raised *right here*, before any
   embedding work happens at all — no wasted computation on a brand that can't be
   scored anyway. The CLI path (`strict=False`) proceeds regardless, and just records
   `scorable: false` in the manifest — that's the CLI's long-standing behavior,
   unchanged, since a developer generating offline may want to inspect an
   under-specified brand's vectors anyway.

2. A **warning** is appended if the generic corpus has fewer than
   `GENERIC_CORPUS_WARNING_THRESHOLD = 15` chunks (`thin_generic_corpus`) — this
   doesn't block anything, it's advisory (the README calls 15-30 chunks "enough" for a
   stable centroid; below that, the warning fires but generation proceeds).

3. **`FITTING_EMBEDDER`** — `TfidfEmbedder().fit_discriminative(brand_texts,
   generic_texts)` — the discriminative reweighting from §1.7, fit once on both
   corpora together.

4. **`ENCODING_VECTORS`** — every brand chunk and every generic chunk gets encoded
   through that now-fitted embedder, producing `brand_vecs` and `generic_vecs`
   (float32, L2-normalized).

5. Everything is written to a **temporary directory first** (`tempfile.mkdtemp(...,
   dir=out_root)`), not directly to the final output folder — explained in §3.3.

6. **`BUILDING_INDEX`** — FAISS `IndexFlatIP` built and written for both vector sets
   (§1.9).

7. **`WRITING_MANIFEST`** — `manifest.json` written (brand_id, brand_name, industry,
   dims, chunk counts, layers present, scorable flag+message, warnings), plus
   `source_input.json` (the raw input dicts, so a later run can be inspected or
   reproduced without needing the original request).

8. **Completeness check** — before doing the atomic swap into place, every expected
   output filename is checked to exist in the temp dir; if anything's missing, a
   `RuntimeError` is raised rather than silently publishing a partial folder.

9. **Atomic swap** — see §3.3.

## 3.3 Why a temp dir + `os.replace`, not writing directly

If a reader (the API's hot-reload path, or a fresh process starting up and scanning
`vector_generation/embeddings/`) tried to read a brand folder *while it was still
being written*, it could see a half-written `manifest.json` or a `brand_vectors.npy`
that doesn't match the `brand_meta.json` sitting next to it — silent corruption, hard
to debug. The fix:

```python
tmp_dir = Path(tempfile.mkdtemp(prefix=f".tmp-{fp.brand_id}-", dir=out_root))
try:
    ... write everything into tmp_dir ...
    out_dir = out_root / fp.brand_id
    old_dir = out_root / f".old-{fp.brand_id}"
    if old_dir.exists():
        shutil.rmtree(old_dir)
    if out_dir.exists():
        out_dir.rename(old_dir)          # move the OLD folder aside, not delete it yet
    try:
        tmp_dir.replace(out_dir)         # atomic rename on the same filesystem
    except Exception:
        if old_dir.exists():
            old_dir.rename(out_dir)      # roll back on failure
        raise
    if old_dir.exists():
        shutil.rmtree(old_dir)
except Exception:
    shutil.rmtree(tmp_dir, ignore_errors=True)
    raise
```

Everything is built fully, in isolation, in a hidden temp directory. Only once every
file is confirmed present does the code touch the real output path — and even then, it
first moves the *existing* folder aside (rather than deleting it) so that if the final
rename somehow fails, the old, known-good folder can be restored. A reader at any
point in time sees either the complete old folder or the complete new one — never a
partial write. This is the same pattern `vector_generation/industries.py` uses for its
own single-file writes (write to a temp file, then `Path(tmp_path).replace(...)`).

## 3.4 `vector_generation/industries.py` — the immutable registry

The generic/competitor corpus is owned by **industry**, not by brand — this is a
deliberate separation so ten language-learning brands don't each need their own copy
of the same competitor corpus. `get_or_create()` is the entire write surface:

```python
def get_or_create(root, industry_id, items):
    with _LOCK:
        existing = _INDUSTRIES.get(industry_id)
        if existing is not None:
            return existing, False           # NEVER overwrites, no matter what `items` says
        if not items:
            raise ValueError(f"Industry '{industry_id}' is not registered.")
        record = {"industry_id": industry_id, "items": items, "chunk_count": len(items)}
        ... write to a temp file, then atomically rename into place ...
        _INDUSTRIES[industry_id] = record
        return record, True
```

Three behaviors worth internalizing:

- **One-way**: an `industry_id`, once created, is never edited by this function again
  — even if a later caller passes different `items` for the same id, they're silently
  ignored (the caller finds out via a warning, not a silent no-op — see
  `_resolve_industry` in Part 6). To change a corpus, you register a *new* id (e.g.
  `outdoor_gear_apparel_v2`).
- **Race-safe creation**: the entire check-then-create sequence is inside one lock
  (`threading.Lock`), so two concurrent first-time requests for the same brand-new
  `industry_id` can't both pass the "doesn't exist yet" check and then both try to
  write — one wins, the other sees `existing is not None` on its turn inside the lock.
- **In-memory + on-disk mirror**: `_INDUSTRIES` is a module-level dict, populated once
  at process startup by `load_all()` (called from `api.py`'s `lifespan`), and updated
  in place whenever a new industry is created. A restart re-reads every `.json` file
  under `vector_generation/industries/` from scratch — the dict itself doesn't need to
  survive a restart because the files do.

## 3.5 `vector_generation/jobs.py` — the state machine for live generation

Purely in-memory bookkeeping (explicitly documented as not needing to survive a
restart, since the actual embeddings it tracks progress toward *do* persist to disk
independently). Two module-level dicts under one lock:

- `_JOBS: dict[job_id, JobRecord]` — the job's current status/stage/warnings/error.
- `_BRAND_LOCKS: dict[brand_id, job_id]` — which job (if any) currently "owns"
  generation for a given brand, so two concurrent submissions for the same brand can't
  race each other.

`JobStatus` moves `queued → running → ready | failed`. While `running`, `JobStage`
walks through the same five stages `generate_from_dicts()` reports via its `on_stage`
callback: `fingerprinting → fitting_embedder → encoding_vectors → building_index →
writing_manifest`. `try_acquire_brand_lock()` / `release_brand_lock()` bracket a job's
lifetime — released on both `complete_job()` and `fail_job()`, so a failed job doesn't
permanently block that brand from being retried.

## 3.6 What actually lands on disk, per brand

Every `vector_generation/embeddings/<brand_id>/` folder contains exactly these files
(and `BrandVectorStore.__init__`, §4.1, will fail to load the folder if any are
missing):

| File | Contents |
|---|---|
| `manifest.json` | brand_id, brand_name, industry, dims, chunk counts, layers present, scorable flag+message, warnings |
| `brand_vectors.npy` / `generic_vectors.npy` | `(N, D)` float32, L2-normalized, one row per chunk |
| `brand_meta.json` / `generic_meta.json` | one `{text, asset_type, layer, words}` object per row, **same order** as the corresponding `.npy` — this is what makes it possible to open the folder and see exactly what got embedded, in what order |
| `vectorizer.joblib` | the fitted `sklearn.TfidfVectorizer` (pickled via joblib) |
| `term_weights.npy` | the discriminative term weights from §1.7, aligned to the vectorizer's vocabulary indices |
| `brand.index` / `generic.index` | FAISS `IndexFlatIP`, one per corpus |
| `source_input.json` | the raw `{brand, generic}` dicts that produced this folder, for later inspection/reproduction |

---

# Part 4 — Runtime Loading

## 4.1 `loci/vector_store.py` — `BrandVectorStore`

This class does exactly one thing: **load**, never generate. `BrandVectorStore.load(folder)`
reads every file from §3.6 into memory once, and holds it there — vectors, FAISS
indices, the fitted `InputEncoder` — for the life of the process. It also
reconstructs `BrandFingerprint` and `GenericCorpus` Python objects from the persisted
meta JSON (`Chunk(**m) for m in self.brand_meta`), so the rest of the runtime code
(`loci/scorer.py`) can work with the same typed objects it would if the fingerprint
had just been built from scratch — nothing downstream needs to know whether the
fingerprint came fresh from `from_assets()` or was reconstructed from a saved folder.

Two families of methods worth understanding, since they encode subtle fallback rules:

**Per-layer vector slicing** — `brand_layer_vectors(layer)` / `generic_layer_vectors(layer)`
filter the full vector arrays down to just the rows whose metadata says they belong to
that layer. `generic_layer_vectors` carries the same "`< 3` chunks → use the whole
corpus instead" rule as `GenericCorpus.texts()` (§2.3) — kept in sync deliberately, so
vector-level and text-level behavior never diverge.

**Calibration subsets** — `brand_calibration_vectors(layer)` / `generic_calibration_vectors(layer)`
implement the `MIN_CALIB_WORDS = 8` floor mentioned in §1.5's anchor table:

```python
@staticmethod
def _calibration_subset(vectors, meta, idx):
    long_idx = [i for i in idx if meta[i]["words"] >= MIN_CALIB_WORDS]
    long_vecs = vectors[long_idx]
    return long_vecs if len(long_vecs) >= 2 else vectors[idx]
```

Very short chunks (a bare brand name, a two-word tagline fragment) have near-zero
cosine similarity to almost everything, purely because there's so little vocabulary in
them to compare — including them in the anchor calculation would drag every anchor
toward zero regardless of how genuinely on-brand or generic the *rest* of the corpus
is. So calibration anchors (§1.5, §1.6) are computed only from chunks with at least 8
words — but the actual brand/generic **centroids** (used for every real scoring
comparison) still use every chunk, short ones included, since the centroid direction
itself isn't distorted by mixing short and long chunks the way a leave-one-out
similarity measurement would be.

**FAISS lookups** — `nearest_brand_chunks()` / `nearest_generic_chunks()`, already
covered in §1.9.

## 4.2 `loci/input_encoder.py` — the one live embedding path

Already covered in depth in §1.2 and §1.8. The one-sentence summary worth repeating:
this class can **load** a fitted vectorizer + term weights and **encode** new text
through them, and that's the entire interface — no `.fit()` exists on this class at
all. That asymmetry (fit lives in `vector_generation/`, encode-only lives in `loci/`)
is what guarantees the live API can never accidentally refit a vector space in
response to a request.

---

# Part 5 — The Scoring Engine, Metric by Metric

## 5.1 `loci/metrics/centroid.py` — `DualCentroidModel`

The mechanical heart of the whole system. One instance is built **per layer present
in the brand's fingerprint** (`BrandDistinctivenessScorer.__init__`, §5.5), each with
its own brand centroid, generic centroid, and calibration anchors — because "what
counts as on-brand" legitimately differs between, say, the Identity layer (mission,
tagline) and the Voice layer (blog posts, ads).

Construction (`__init__`, `centroid.py:57-80`):

1. Requires at least 2 vectors in both the brand and generic vector sets — raises
   `ValueError` otherwise, since a centroid from a single point isn't meaningful and
   leave-one-out calibration needs at least 2 points to leave one out from.
2. Picks calibration subsets — falls back to the full vector set if fewer than 2
   calibration-eligible (§4.1's `MIN_CALIB_WORDS` filter) vectors are available for
   either side.
3. Computes `brand_centroid` and `generic_centroid` (§1.4) from the **full** vector
   sets (not the calibration subsets — see §4.1's explanation of why these two things
   use different inputs).
4. Calls `_calibrate()` to compute the four anchors from §1.5, using leave-one-out
   similarity (§1.6) for the two "self" anchors.

`score(text)` (`centroid.py:106-122`) — the only method called per request:

1. Encodes `text` through the shared `InputEncoder` (§4.2) — **this happens once per
   layer**, since each `DualCentroidModel` holds a reference to the same encoder
   object but calls `.encode()` independently. See Part 8 for why this redundancy is
   harmless in practice.
2. Computes raw cosine similarity to both centroids: `sb` (vs. brand) and `sg` (vs.
   generic).
3. Applies the coverage discount (§1.8) to `sb` only.
4. Rescales both (§1.5) against this layer's own calibration anchors, returning a
   `CentroidScores(consistency, distinctiveness, raw_sim_brand, raw_sim_generic)` —
   the raw similarities are kept around purely for display in the API response's
   `evidence.raw_cosine` block, not used again downstream.

## 5.2 `loci/metrics/lexical.py` — `LexicalScorer`

Already covered in depth in §1.10. One instance per scorer, built once from **all**
brand texts vs. **all** generic texts (not per-layer, unlike the centroid model) —
`signatures` and `cliches` are brand-wide vocabulary sets. `.score(text)` is called
once per request (not once per layer) and its output (`signature_hit`, `cliche_free`,
`diversity`) is reused, with different weights, across every layer's blend in
`scorer.py` — see §5.5.

## 5.3 `loci/metrics/structural.py` — `StructuralScorer`

Already covered in depth in §1.11. Same pattern as lexical: one instance, built once
from all brand texts, `.score(text)` called once per request, its `style_match` value
reused across whichever layers' weight tables include a `structural` term (Messaging
and Voice — see the weight table in §5.5; Identity and Positioning don't use it at
all).

## 5.4 `loci/metrics/tone.py` — `ToneScorer`, and the call-economy layer around it

The mechanics of tone judging (heuristic formulas, the LLM prompt, why an LLM is used
at all) are covered in §1.12. This section covers the class that wraps it, and the
caching/budget/tracking layer built around it (added specifically to make repeated
`/score` calls cheap and to bound worst-case Anthropic API spend on a public,
unauthenticated demo).

**`ToneScorer.__init__(brand_voice_exemplars)`** (`tone.py`, class body near the end of
the file) — profiles up to 5 voice-layer exemplars once, at brand-load time (called
from `BrandDistinctivenessScorer.__init__`, §5.5, with `voice_exemplars[:5]`, where
`voice_exemplars` is the brand's Voice-layer texts, or the first 5 texts overall if the
brand has no Voice-layer assets). Averages each of the 5 axes across all profiled
exemplars to get one `brand_profile` dict. `self.judge` records `"llm"` if *any*
exemplar was judged by the LLM, `"heuristic"` otherwise — so a brand's overall judge
label reflects whether real LLM judging was available at all during profiling, even if
individual exemplars happened to hit a transient error and fell back individually.

**`ToneScorer.score(text)`** — called once per `/score` request (`scorer.py:139`):
profiles the input text the same way, computes the absolute gap between the input's
profile and the brand's profile on each of the 5 axes, averages the gaps, and converts
that average gap (on a 0–10 scale, since axes are rated 0–10) into a 0–100 alignment
score:

```python
gaps = {a: abs(p[a] - self.brand_profile[a]) for a in AXES}
mean_gap = sum(gaps.values()) / len(gaps)
alignment = round(max(0.0, 1 - mean_gap / 4.5) * 100, 1)
```

A mean gap of 0 (identical axis ratings) → 100. A mean gap of 4.5 or more (out of a
possible 10) → 0. `biggest_gap` records which single axis contributed the most
disagreement — this is what shows up as `tone_biggest_gap` in the API response, useful
for a human to see *what kind* of tone mismatch was detected, not just that one
occurred.

**The call-economy layer** (this is the part built specifically for this deployment's
cost/traceability needs, and is a good example of "keep the hard part, cut the
redundant calls"):

- **Exact-text cache.** `profile(text)` is a pure function of its input, so results are
  cached keyed by `sha256(text.strip().lower())`, bounded at 2,000 entries (oldest
  evicted once full). This catches the most likely real-world repeat pattern —
  re-scoring the same draft, or someone hammering the rate limit with one repeated
  payload — without needing any fuzzy-matching complexity. It has zero effect on
  correctness: identical input always gets an identical judged (or heuristic) profile
  anyway, so caching it changes nothing about what score comes back, only how often an
  API call is made to produce it.
- **Daily call budget.** `ANTHROPIC_CALL_BUDGET` (env var, default `200`) caps how many
  *actual* LLM calls this process will attempt per UTC day, tracked by the
  `ToneUsage.llm_calls` counter. Once the budget's exhausted, every subsequent request
  falls back to the heuristic — silently, the same way a missing API key or a network
  error would — until the counter rolls over at UTC midnight. This exists because
  per-IP rate limiting (Part 7) bounds *one visitor's* worst case, but not the
  *aggregate* worst case across many different visitors on a public, unauthenticated
  demo; the budget is the actual dollar-cost backstop.
- **`ToneUsage`** — a small thread-safe counter object (`usage`, a module-level
  singleton) tracking `llm_calls`, `llm_errors`, `cache_hits`, `heuristic_fallbacks`,
  and `budget_exhausted`, all resetting at UTC midnight. `.snapshot()` returns the
  current counts as a plain dict, exposed live via `GET /admin/tone-usage` (Part 6).
  Every real LLM call also logs one line (model name + input/output token counts) via
  the standard `logging` module, so Render's log viewer shows real-time cost signal
  even without hitting the usage endpoint.

The full decision path inside `profile(text)`, in order:

1. Cache hit on the exact normalized text? → return it immediately, record
   `cache_hits`.
2. Under budget? → attempt `_llm_profile(text)`. If it returns a real profile → cache
   and return `("llm", profile)`.
3. Otherwise (over budget, no API key, or the LLM call errored) → fall back to
   `_heuristic_profile(text)`, recording either `budget_exhausted` or
   `heuristic_fallbacks` depending on which case it was, then cache and return.

## 5.5 `loci/scorer.py` — `BrandDistinctivenessScorer`, the aggregator

This is the module that ties every metric above into one `Report`. Nothing in this
class computes a metric itself — it only builds one instance of each scorer, calls
each once (or once per layer, for the centroid model), and combines the results with
fixed weights.

**Construction** (`__init__`, `scorer.py:98-127`):

- One `DualCentroidModel` per layer actually present in the fingerprint
  (`self.fp.layers_present()`) — if a layer has fewer than 2 brand vectors, a warning
  is recorded (`"Layer '{layer}' has <2 assets — falling back to full corpus."`) and
  the model is built against the *entire* brand vector set instead of just that
  layer's slice, so a thin layer still gets *a* comparison, just a less specific one.
- One `LexicalScorer` over **all** brand texts vs. **all** generic texts (not
  per-layer).
- One `StructuralScorer` over **all** brand texts.
- One `ToneScorer` over up to 5 Voice-layer exemplars (or the first 5 texts overall,
  if the brand has no Voice-layer assets at all).

**The weight table** — `LAYER_WEIGHTS` (`scorer.py:28-49`) is the single place that
decides, per layer, how much each underlying metric contributes to that layer's
consistency and distinctiveness scores:

| Layer | Consistency | Distinctiveness |
|---|---|---|
| **Identity** | centroid .60, signature .20, tone .20 | centroid .60, cliché .40 |
| **Messaging** | centroid .40, signature .20, structural .15, tone .25 | centroid .45, cliché .40, diversity .15 |
| **Voice** | centroid .20, signature .20, structural .25, tone .35 | centroid .30, cliché .45, diversity .25 |
| **Positioning** | centroid .70, signature .30 | centroid .55, cliché .45 |

Two patterns worth noticing: **consistency always includes `tone` (or drops it for
Positioning, which has no tone weight at all) and always includes `centroid`**; **every
distinctiveness column is centroid + cliché, with `diversity` (lexical MSTTR) added
only for Messaging and Voice**. And structurally, `centroid`'s weight *decreases* from
Identity (.60) down to Voice (.20) while `tone`+`structural` (style-based signals)
*increase* — Identity is short, declarative, semantic text (a tagline, a mission
statement) where vector similarity is the strongest signal available; Voice is
free-form prose (blog posts, ads) where *how* something is said carries more weight
than *what* topic it covers, echoing the exact reasoning from §1.12 about embeddings
encoding topic more than tone.

**`_blend(parts, weights)`** (`scorer.py:89-95`) — the function that actually applies
these weights:

```python
def _blend(parts, weights):
    avail = {k: w for k, w in weights.items() if parts.get(k) is not None}
    total = sum(avail.values()) or 1.0
    score = sum(parts[k] * (w / total) for k, w in avail.items())
    detail = {k: {"value": round(parts[k], 1), "weight": round(w / total, 2)}
              for k, w in avail.items()}
    return round(score, 1), detail
```

It re-normalizes weights over whichever parts are actually present (`parts.get(k) is
not None`) — in current usage every part is always present for every configured
layer, so this re-normalization is dormant, but it's the mechanism that would let a
future metric be optional (e.g. skip `tone` for a brand with no voice exemplars)
without needing every weight table hand-edited to compensate.

**`score(copy)`** (`scorer.py:135-211`), the full per-request path:

1. Compute `lex`, `struct`, `tone` **once**, for the input text (not per layer — these
   three scorers don't vary by layer).
2. For each layer present that also has an entry in `LAYER_WEIGHTS` (silently skipping
   any layer that isn't scored, e.g. Proof):
   - `cen = model.score(text)` — the layer-specific centroid consistency/distinctiveness
     pair (§5.1) — this **does** vary per layer, since each layer has its own
     `DualCentroidModel`.
   - Blend consistency from `{centroid: cen.consistency, lexical_sig: lex.signature_hit,
     structural: struct.style_match, tone: tone.alignment}` against this layer's
     weights.
   - Blend distinctiveness from `{centroid: cen.distinctiveness, lexical_cliche:
     lex.cliche_free, lexical_div: lex.diversity}` against this layer's weights.
   - Look up the quadrant (§1.13) from `(cons >= 55, dist >= 55)`.
   - Record a `LayerVerdict` with both scores, the quadrant, and a `contributions`
     dict (per-metric value + normalized weight, plus the raw, un-rescaled cosine
     similarities — purely for transparency/debugging in the response).
3. **Aggregate to an overall verdict** via `agg(attr)`:

   ```python
   def agg(attr):
       num = den = 0.0
       for v in verdicts:
           w = 2.0 if v.layer == copy.intended_layer.value else 1.0
           num += getattr(v, attr) * w
           den += w
       return round(num / max(den, 1), 1)
   ```

   Every scored layer contributes, but the layer matching `copy.intended_layer`
   (default `MESSAGING`) counts **twice** — this is the mechanism behind the README's
   "`intended_layer` doubles that layer's weight" note (§2.4). Every other present
   layer counts once. The overall quadrant is then looked up from the aggregated
   `(oc, od)` pair, using the same 55.0 threshold.

4. **One more encode**, separate from every per-layer `DualCentroidModel.score()`
   call: `self.store.encoder.encode([text])[0]`, used purely to query both FAISS
   indices for the "nearest evidence" chunks (§1.9) — `nearest_brand_chunks` and
   `nearest_generic_chunks`. This is a genuinely redundant encode (the exact same text
   was already encoded once inside every layer's `DualCentroidModel.score()` call), but
   TF-IDF-transforming one short string is cheap enough that the redundancy has no
   meaningful cost — see Part 8 for the exact call count on a real request.
5. Assemble and return the `Report` dataclass: brand name, input label, overall
   consistency/distinctiveness/quadrant/note, the full list of per-layer `LayerVerdict`s,
   an `evidence` dict (signature terms used, clichés detected, lexical diversity,
   structural per-feature detail, tone input/brand profiles + biggest gap + judge type,
   both FAISS nearest-chunk lists), and any accumulated `warnings` (carried over from
   `BrandVectorStore`, e.g. a thin-layer fallback notice).

`Report.to_dict()` is just `dataclasses.asdict(self)` — this is the exact JSON shape
`POST /brands/{id}/score` returns (Part 6).

---

# Part 6 — The API, Endpoint by Endpoint (`api.py`)

## 6.0 App setup

- **Lifespan** (`lifespan()`, `api.py:89-94`) — on startup, loads every registered
  industry from disk (`industries.load_all`) and then every brand folder under
  `vector_generation/embeddings/` (`_load_all_brands()`), building one
  `BrandDistinctivenessScorer` per brand and holding all of them in the module-level
  `_SCORERS: dict[brand_id, BrandDistinctivenessScorer]`. On shutdown, `_SCORERS` is
  cleared (mostly symbolic for a process that's about to exit anyway). No request ever
  triggers this loading path again — new brands only enter `_SCORERS` via the live
  embeddings endpoint (§6.7), which updates the dict in place, no restart required.
- **Rate limiter** — a single `slowapi.Limiter` keyed by remote IP address
  (`get_remote_address`), registered as `app.state.limiter` with `RateLimitExceeded`
  wired to slowapi's default 429 handler. Individual limits are set per-endpoint via
  `@limiter.limit(...)` decorators — covered per endpoint below.
- **CORS** — off by default (fails closed: a browser calling this API from another
  origin gets no `Access-Control-Allow-Origin` header, so the request is blocked
  browser-side). Set via `ALLOWED_ORIGINS` (comma-separated) in the deployment's
  environment; once set, `CORSMiddleware` is added, allowing `GET`/`POST` and all
  headers from exactly those origins. Until the frontend's real deployed domain is
  known, this stays empty and cross-origin browser calls simply don't work — same-origin
  calls, `curl`, and server-to-server calls are unaffected either way (CORS is a
  browser-enforced restriction, not a server-side auth check).
- **Disclaimer** — a fixed string (`DEMO_DISCLAIMER`) surfaced both in `GET /`'s JSON
  and in the FastAPI app's `description` (which renders in the auto-generated
  `/docs` page) — stating plainly that this is a temporary, in-memory deployment and a
  restart/redeploy/inactivity-spindown wipes anything created live.

## 6.1 `GET /`

Returns the service name, a flat list of every capability (method + path), the
disclaimer text, and a pointer to `/docs`. No side effects, no rate limit — this is
also the configured Render health-check path (`render.yaml`'s `healthCheckPath: /`).

## 6.2 `GET /brands`

Lists every brand currently held in `_SCORERS` — `brand_id`, `brand_name`, and any
accumulated `warnings` (from `BrandVectorStore`, e.g. thin-layer fallbacks). No rate
limit; read-only, cheap (just a dict iteration, no computation).

## 6.3 `POST /brands` — mint a brand_id (rate limit: 20/hour per IP)

This endpoint does **not** create a scorable brand — it only reserves a unique
`brand_id` string, derived from a slugified `brand_name` plus a random 6-hex-character
suffix (`_mint_brand_id`, `api.py:167-173`), checked against both the in-memory
`_SCORERS` dict and the on-disk `vector_generation/embeddings/` folder so a freshly
minted id can never collide with an existing brand either in memory or on disk (up to
20 retry attempts before giving up with a `500`). The returned `brand_id` is meant to
be used as the `{brand_id}` path parameter on a subsequent `POST
/brands/{brand_id}/embeddings` call — it's a name-reservation step, separated from
actual generation, presumably so a frontend can show/reference a brand's id before the
(potentially slow) embedding job has run.

## 6.4 `GET /brands/{brand_id}`

`_get_scorer(brand_id)` (`api.py:120-128`) is the shared lookup used by every endpoint
that needs an existing brand — raises `404` with a message pointing at both ways to
create the brand (`POST /brands/{id}/embeddings` or the offline CLI + restart) if the
id isn't in `_SCORERS`. On success, returns brand name, industry, MVBF status (met +
which fields are missing, if any), which layers are present, the `scorable` flag +
message from the manifest, every learned signature/cliché term (full list, not
truncated — contrast with `evidence.matched_signature`/`matched_cliche` in a score
response, which cap at 12), and accumulated warnings.

## 6.5 `POST /brands/{brand_id}/score` — the main endpoint (rate limit: 60/minute per IP)

Request body (`ScoreRequest`): `{text, intended_layer="messaging", channel="landing_page",
label=None}`. Looks up the scorer, builds an `InputCopy` from the request fields, and
returns `scorer.score(copy).to_dict()` directly — the full `Report` shape traced in
§5.5 and demonstrated with real numbers in Part 8. This is the only endpoint that
triggers live computation on every call (every other endpoint either reads
already-computed state or kicks off a background job) — every metric in Part 5 runs
fresh, on this exact input text, every time this endpoint is hit. It's also the only
endpoint that can trigger a live LLM call (via `ToneScorer.score()`, §5.4), subject to
the cache and daily budget described there.

## 6.6 `GET /brands/{brand_id}/vocabulary`

A read-only subset of `GET /brands/{brand_id}`'s response — just `brand_id`,
`signature_terms`, `cliche_terms`, both full lists. No rate limit; no computation
(these sets were built once, at scorer construction time).

## 6.7 `POST /brands/{brand_id}/embeddings` — live generation (rate limit: 5/hour per IP)

This is the live equivalent of running the CLI generator, triggered by an API call
instead. Two guardrails fire before anything else happens:

1. **`_JOB_SLOTS = threading.Semaphore(2)`** — a process-wide cap on how many
   embedding-generation jobs can run *concurrently*, regardless of which IP(s) they
   came from. Rate limiting alone only bounds one caller's request rate; it doesn't
   stop five different IPs from each submitting one expensive job at the same moment
   and pegging the CPU together. If both slots are taken, the request is rejected
   immediately with `429` (`"Server is at capacity..."`) rather than queued — the
   caller is expected to retry later, not wait.
2. **Per-brand lock** (`jobs.try_acquire_brand_lock`) — if a job is already in flight
   for this specific `brand_id`, the new request gets `409 Conflict` (and releases the
   semaphore slot it had just acquired, so it doesn't leak a slot for a request that's
   being rejected).

Once past both guards: a `job_id` (UUID4) is minted, a `JobRecord` created in
`QUEUED` state, and the actual work (`_run_embedding_job`) is handed to FastAPI's
`BackgroundTasks` — the HTTP response (`202 {job_id, status: "queued"}`) returns
immediately, before generation has even started.

**`_run_embedding_job`** (`api.py:252-292`), run in the background:

1. `jobs.start_job(job_id)` → status `RUNNING`.
2. **Resolve the industry** (`_resolve_industry`) — calls
   `industries.get_or_create(...)` (§3.4). If the industry isn't registered and no
   `items` were supplied, this raises `ValueError`, caught here and turned into a
   `unknown_industry` job failure. If the industry *does* already exist and `items`
   was also sent, a non-fatal `industry_corpus_ignored` warning is attached to the job
   (generation still proceeds, using the existing registered corpus).
3. **Generate** — calls the exact same `generate_from_dicts(..., strict=True)` from
   §3.2, with `on_stage` wired to `jobs.update_stage` so the job's `stage` field
   updates live as generation progresses. `strict=True` here is what turns an
   unscorable brand (§2.2) into a hard `mvbf_not_met` job failure *before* anything is
   written to disk or hot-loaded — unlike the CLI path, this live path cannot produce
   a "successfully generated but not actually scorable" brand.
4. Any warnings from the manifest (e.g. `thin_generic_corpus`) are copied onto the job
   record.
5. **Hot-load** — `BrandVectorStore.load(out_dir)` then `BrandDistinctivenessScorer(store)`,
   assigned directly into `_SCORERS[brand_id]` — this is the entire "no restart
   needed" mechanism. The very next `POST /brands/{brand_id}/score` call after this
   line executes will use the new scorer.
6. `jobs.complete_job(job_id)` → status `READY`, brand lock released.
7. **`finally: _JOB_SLOTS.release()`** — guarantees the concurrency slot is freed
   whether the job succeeded, failed with a known error, or crashed with something
   unexpected (the bare `except Exception` branches around fingerprinting/industry
   resolution and around generation both log the full traceback via
   `logger.exception(...)` and record an `internal_error` job failure, rather than
   letting the background task die silently).

## 6.8 `GET /jobs/{job_id}`

Returns the raw `JobRecord` (§3.5) — `status`, `stage` (while running), accumulated
`warnings`, `error` (with `code`/`message`/`fields` if failed), and
created/started/completed timestamps. `404` if the id was never issued. This is the
polling endpoint a frontend would loop on after a `202` from §6.7.

## 6.9 `GET /industries`

Lists every registered industry (`industry_id`, `chunk_count`) alongside which
currently-loaded `brand_id`s reference it — computed by scanning `_SCORERS` for each
brand's `manifest["industry"]`, so this reflects brands **currently in memory**, not
every brand that's ever referenced that industry historically (a brand whose folder
exists on disk but wasn't loaded — e.g. if `vector_generation/embeddings/` was
manually pruned before the last restart — wouldn't show up here even if its manifest
still references the industry).

## 6.10 `GET /industries/{industry_id}`

Returns the raw stored record (`industry_id`, `items`, `chunk_count`) for one
industry. `404` if unregistered. There is deliberately no `PUT`/update endpoint here —
industries are immutable by design (§3.4).

## 6.11 `GET /admin/tone-usage`

Added alongside the caching/budget layer in §5.4 — returns `tone_metrics.usage.snapshot()`
directly: `date_utc`, `daily_budget`, and the five running counters (`llm_calls`,
`llm_errors`, `cache_hits`, `heuristic_fallbacks`, `budget_exhausted`). Purely
read-only telemetry, no side effects, no rate limit currently applied. Since counters
live in the same in-memory space as everything else, this resets on every restart —
consistent with the rest of this deployment's "nothing here is meant to survive a
restart except what's on disk" philosophy.

## 6.12 `GET /schema/assets`

A static, computed-from-code reference endpoint — inverts `ASSET_LAYER_MAP` (Part 2)
into `{layer: [asset_types]}`, alongside the six `MVBF_FIELDS` and the list of layers
that are actually scored (`sorted(LAYER_WEIGHTS.keys())` — i.e. excludes `proof`, per
the note in §2.1). Exists so a frontend or API consumer can discover valid asset types
and MVBF requirements without hardcoding them separately from the backend.

---

# Part 7 — Cross-Cutting Concerns

## 7.1 Rate limiting and concurrency

Three independent guardrails, each protecting against a different failure mode:

| Mechanism | Scope | Protects against |
|---|---|---|
| `@limiter.limit(...)` (slowapi, per-IP) | `POST /brands` (20/hour), `POST /brands/{id}/score` (60/minute), `POST /brands/{id}/embeddings` (5/hour) | one visitor hammering an endpoint |
| `_JOB_SLOTS = Semaphore(2)` | `POST /brands/{id}/embeddings` | many *different* IPs each submitting one expensive job simultaneously — per-IP limits alone don't catch this |
| `ANTHROPIC_CALL_BUDGET` (default 200/day, process-wide) | tone judging specifically, inside `/score` | many different visitors, in aggregate, driving real dollar cost on an unauthenticated public endpoint |

No authentication exists anywhere in this API — a deliberate choice for a temporary,
public demo (documented in earlier project decisions): these three mechanisms are the
entire guardrail, in place of API keys, so the demo stays freely explorable.

## 7.2 LLM call economics

Covered in full in §5.4 and §6.11. The one-sentence version: the LLM is used for
exactly one thing (tone judging), it's the only network call anywhere in the scoring
path, it's gated behind `ANTHROPIC_API_KEY` (currently **unset** on the Render
deployment — see `render.yaml`, which has no `ANTHROPIC_API_KEY` entry — so this
deployment runs the heuristic tone fallback exclusively, right now), and if the key is
ever added, the cache + daily budget + usage endpoint are already in place to bound
and observe the resulting cost from day one.

## 7.3 Deployment

- **`render.yaml`** — a Render Blueprint: `pip install -r requirements.txt` build step,
  `uvicorn api:app --host 0.0.0.0 --port $PORT` start command, health check at `/`,
  `PYTHON_VERSION` pinned to `3.12.9`, `ALLOWED_ORIGINS` defaulted to empty (CORS off
  until explicitly configured). **No autoscaling/multi-instance block, on purpose** —
  a comment in the file spells out why: brand state (`_SCORERS`), job status
  (`vector_generation/jobs.py`), the industries registry, rate-limit counters, and the
  tone-usage counters are **all in-process memory**. Running more than one instance
  would mean each instance has its own independent, diverging view of all of that
  state — a brand created via the live endpoint on instance A would be invisible to
  instance B, rate limits would be bypassable by hitting different instances, etc.
  Staying single-instance, single-worker is a hard requirement of the current
  architecture, not a cost-saving choice.
- **`.python-version`** — pins `3.12.9`, matching the pinned `PYTHON_VERSION` env var
  in `render.yaml`, so a local dev environment and Render's build use the identical
  interpreter version.
- **`requirements.txt`** — every dependency pinned to an exact version (not `>=`),
  taken from a known-working local environment, specifically so a fresh Render build
  can never silently drift onto a newer, untested version of any dependency. The two
  commented-out lines (`sentence-transformers`, `anthropic`) mark the two optional
  upgrade paths (§1.1's dense-embedding alternative, and the LLM tone judge) that
  aren't installed by default — `anthropic` is imported lazily, inside a `try/except`,
  precisely so its absence doesn't break anything; it just means `_llm_profile()`
  always fails and falls back to the heuristic.

## 7.4 Known limitations (carried from `README.md`, restated here for completeness)

- Positioning is scaffolded (has a weight table entry, §5.5) but only meaningful with
  a real, scraped competitor corpus behind it.
- The generic centroid is only as good as the competitor list fed into it — a bad or
  unrepresentative list produces a confidently wrong distinctiveness score; nothing in
  the system can detect that the *input* corpus itself was poorly chosen.
- Every threshold in this system (`THRESHOLD = 55.0`, MSTTR's `0.75` normalizer, the
  `0.25`/`0.30` lexical saturation points, the `2.5` z-score ceiling, the `4.5` tone
  gap ceiling, the anchor percentiles `85/60/40/60`) is **hand-tuned on one brand**
  (Duolingo), not learned from data or validated across many brands.
- No scraper exists — brand/competitor assets are supplied as hand-written JSON.
- The tone profile is computed at process startup (inside `ToneScorer.__init__`),
  **not** precomputed and persisted by `vector_generation/` the way vectors are — a
  brand's tone axes are a judgment, not a vector, so this fell outside the scope of
  that refactor and is recomputed (subject to the cache in §5.4) every time a brand is
  loaded, rather than being a fixed artifact on disk.
- Job state and the industries registry are single-process, in-memory constructs —
  they do not coordinate across instances and do not need to (§7.3's single-instance
  requirement), but this means they would need real infrastructure (a queue, a
  database) to ever scale past one process.
- Industries are immutable by design (§3.4) — there is no way to fix a typo or bad
  entry in an already-registered industry's corpus in place; the only path is
  registering a new `industry_id`.

---

# Part 8 — A Complete Worked Trace, With Real Numbers

Everything below is copied from an actual run of `python demo.py` against this exact
repository's `data/brand_duolingo.json` fingerprint and `data/generic_edtech.json`
generic corpus — not invented numbers.

## 8.1 Setup

`data/brand_duolingo.json` supplies 12 of the 22 possible asset types (all 6 MVBF
fields plus `founder_story`, `homepage`, `product_page`, `cta`, `blog`, `social`,
`ad`, `job_post`, `support_doc`, `case_study`, `testimonial`), producing **30 chunks
across 4 layers** (Identity, Messaging, Voice, Proof — Proof exists in the fingerprint
but, per §2.1, is never scored). `data/generic_edtech.json` supplies **18 generic
chunks** for the `language_learning_edtech` industry. Loading this into a
`BrandDistinctivenessScorer` builds:

- 3 `DualCentroidModel`s (Identity, Messaging, Voice — the layers that are both
  *present* in the fingerprint and *scored*, per `LAYER_WEIGHTS`).
- 1 `LexicalScorer`, over all 30 brand chunks vs. all 18 generic chunks. Learned
  signature terms include (first 14 alphabetically) `actually, answer, bring, built,
  everyone, feel, free, game, games, gems, homework, isn't, keep, lessons`. Learned
  cliché terms include `achieve, across, adaptive, comprehensive, contact, dedicated,
  delivers, driving, enterprise, faster, join, language, leading, learners`.
- 1 `StructuralScorer`, over all 30 brand chunks.
- 1 `ToneScorer`, over up to 5 Voice-layer exemplars (`blog`, `social`, `ad`,
  `job_post`, `support_doc` chunks). In this run, no `ANTHROPIC_API_KEY` was set, so
  every axis profile — both the brand's and every input's — came from
  `_heuristic_profile()`, and `tone_judge` reads `"heuristic"` throughout.

## 8.2 Scoring candidate A — "written in-voice"

Input text (`intended_layer=messaging`):

> "Your streak is 47 days old. Do not let a Tuesday kill it.
> Five minutes. One lesson. The owl is watching and the owl is free.
> Start now. It costs nothing, because it never has."

**Step-by-step, following `BrandDistinctivenessScorer.score()` from §5.5:**

1. `lex = self.lexical.score(text)`, `struct = self.structural.score(text)`, `tone =
   self.tone.score(text)` — computed once, reused for every layer below.
2. For each of the 3 present-and-scored layers, `model.score(text)` runs — this
   re-encodes the exact same input text through the shared `InputEncoder` three
   separate times (once per `DualCentroidModel`), plus a fourth time later for the
   FAISS lookup (§5.5, step 4) — **4 total TF-IDF encode calls for this one request**,
   all cheap.

**Identity layer** — real output:

```json
{
  "layer": "identity",
  "consistency": 93.2,
  "distinctiveness": 91.1,
  "quadrant": "IDEAL",
  "contributions": {
    "consistency": {
      "centroid":    {"value": 100.0, "weight": 0.6},
      "lexical_sig": {"value": 100.0, "weight": 0.2},
      "tone":        {"value": 66.0,  "weight": 0.2}
    },
    "distinctiveness": {
      "centroid":       {"value": 100.0, "weight": 0.6},
      "lexical_cliche": {"value": 77.8,  "weight": 0.4}
    },
    "raw_cosine": {"vs_brand": 0.1515, "vs_generic": 0.0003}
  }
}
```

Read this row by row against §5.5's `_blend()`: consistency = `100.0×0.6 +
100.0×0.2 + 66.0×0.2 = 60 + 20 + 13.2 = 93.2` ✓ — matches exactly. The raw cosine to
the brand centroid is a modest `0.1515` (recall §1.5: raw cosines are small and
meaningless on their own), but it rescales to a perfect `100.0` because it landed at
or above this layer's `cons_high` calibration anchor (the 85th percentile of the
brand's own leave-one-out self-similarity, §1.6) — this input's brand-centroid
similarity is at least as strong as what a typical real brand chunk achieves against
itself. The raw cosine to the generic centroid is nearly zero (`0.0003`), which
rescales to `100.0` distinctiveness — essentially no resemblance to the generic
baseline at all.

**Messaging layer** (the input's `intended_layer`, so this one counts double in the
overall aggregation):

```json
{
  "layer": "messaging",
  "consistency": 89.7,
  "distinctiveness": 90.7,
  "contributions": {
    "consistency": {
      "centroid": {"value": 100.0, "weight": 0.4},
      "lexical_sig": {"value": 100.0, "weight": 0.2},
      "structural": {"value": 87.9, "weight": 0.15},
      "tone": {"value": 66.0, "weight": 0.25}
    },
    "distinctiveness": {
      "centroid": {"value": 99.0, "weight": 0.45},
      "lexical_cliche": {"value": 77.8, "weight": 0.4},
      "lexical_div": {"value": 100.0, "weight": 0.15}
    }
  }
}
```

Check: consistency = `100×0.4 + 100×0.2 + 87.9×0.15 + 66×0.25 = 40 + 20 + 13.185 + 16.5
= 89.685` → rounds to `89.7` ✓. Note `tone`'s raw `alignment` value of `66.0` is
identical in both the Identity and Messaging breakdowns above — that's because `tone`
is computed **once** per request (step 1), not per layer; only its *weight* changes
between layers (0.2 for Identity, 0.25 for Messaging, 0.35 for Voice per §5.5's
table), which is exactly the "reused with different weights" behavior described there.

**Voice layer**: `consistency 85.1`, `distinctiveness 90.0`, `IDEAL` — omitted in full
for brevity, but follows the identical pattern with Voice's weight table (centroid
.20, signature .20, structural .25, tone .35 for consistency).

**Overall aggregation** (`agg()`, §5.5, `intended_layer="messaging"` counts double):

```
overall_consistency  = (93.2×1 + 89.7×2 + 85.1×1) / 4 = 357.7 / 4 = 89.425 → 89.4  ✓
overall_distinctiveness = (91.1×1 + 90.7×2 + 90.0×1) / 4 = 362.5 / 4 = 90.625 → 90.6 ✓
```

Both real numbers from the actual run. `oq = QUADRANTS[(89.4 >= 55, 90.6 >= 55)] =
QUADRANTS[(True, True)] = "IDEAL"`.

**Evidence block** — `signature_terms_used: ["free", "minutes", "nothing", "streak"]`
(these are content words from the input that also appear in the brand's learned
signature set), `cliches_detected: ["start"]` (the word "start" happens to be a
learned cliché term for this brand/industry pair, despite this being the strongest
on-brand candidate — a single cliché hit doesn't sink an otherwise strongly
on-brand/distinctive score, since it's one term among the input's whole vocabulary,
feeding into `cliche_free` at roughly a 1/N rate rather than being a hard veto),
`tone_biggest_gap: "technical_accessible"` (of the 5 tone axes, this is where the
input's heuristic profile diverged most from the brand's own voice-exemplar profile —
even for the highest-scoring candidate, no axis match is ever perfect),
`nearest_brand_chunks` — literally 3 real chunks from Duolingo's own corpus that this
input's vector landed closest to via FAISS (e.g. *"Your streak is 3 days old. Do not
let it die on our watch."* — visibly, thematically almost the same sentence as the
scored input, which is exactly the kind of concrete evidence this lookup is meant to
surface).

## 8.3 The other three candidates, contrasted

| Candidate | Consistency | Distinctiveness | Quadrant |
|---|---|---|---|
| A — written in-voice | 89.4 | 90.6 | **IDEAL** |
| B — right claims, category language | 82.1 | 14.2 | **ON-BRAND BUT GENERIC** |
| C — distinctive, not this brand | 43.3 | 63.7 | **UNIQUE BUT OFF-BRAND** |
| D — neither | 16.0 | 14.6 | **LOST** |

Candidate **B**'s consistency (82.1) is still high — it's true, on-topic, in-category
language ("learn a language for free in 5 minutes a day") — but its distinctiveness
collapses to 14.2, because that phrasing overlaps heavily with the generic corpus (its
evidence block shows 8 detected clichés: `join, language, learners, learning,
platform, start, today, worldwide`, and its nearest-generic-chunk matches are close
paraphrases of the input). This is Part 0's "on-brand but generic" failure mode,
concretely demonstrated by real numbers.

Candidate **C** inverts it: consistency drops to 43.3 (this really doesn't sound like
Duolingo — formal, no contractions, no playful voice) while distinctiveness holds at
63.7 (it doesn't sound like generic ed-tech boilerplate either — it sounds like a
totally different kind of institution). Its per-layer breakdown is the most
interesting of the four: Identity scores `UNIQUE BUT OFF-BRAND` (cons 28.7, dist 91.3
— wildly off-brand at the identity level) while Messaging lands exactly on the
boundary at `ON-BRAND BUT GENERIC` (cons 57.5, dist 49.7) — a real example of a single
piece of copy landing in *different quadrants at different layers*, which is precisely
why this system reports layers independently instead of only an overall verdict.

Candidate **D** ("comprehensive solutions... enterprise offerings...") scores low on
both axes across every layer — it is simultaneously generic *and* off-brand, landing
in `LOST` everywhere, with 5 detected clichés (`comprehensive, contact, enterprise,
solutions, team`) and zero signature terms.

---

# Appendix A — Glossary

- **TF-IDF** — Term Frequency × Inverse Document Frequency; a way to weight words in a
  document by how much they define that document relative to a larger corpus. §1.2.
- **Cosine similarity** — the cosine of the angle between two vectors; on L2-normalized
  vectors, equal to their plain dot product. §1.3.
- **L2 normalization** — scaling a vector to length 1 by dividing by its Euclidean
  norm. §1.3.
- **Centroid** — the mean-pooled, re-normalized "average direction" of a set of
  vectors. §1.4.
- **Calibration / anchors** — rescaling a raw similarity score against two
  empirically-observed reference points (drawn from the corpora themselves) so the
  result is interpretable as 0–100. §1.5.
- **Leave-one-out (LOO)** — measuring a data point against a statistic (here, a
  centroid) computed *excluding* that point, to avoid self-inflation. §1.6.
- **Discriminative reweighting / lift** — down-weighting vocabulary that's common to
  both the brand and its competitors, so similarity reflects brand voice rather than
  shared topic. §1.7, §1.10.
- **Coverage discounting** — reducing a similarity score in proportion to how much of
  the input text was actually recognized by the fitted vocabulary (a TF-IDF-specific
  correction for out-of-vocabulary words). §1.8.
- **FAISS / `IndexFlatIP`** — an exact (non-approximate), inner-product-ranked
  nearest-neighbor search index; inner product equals cosine similarity here because
  every vector is pre-normalized. §1.9.
- **MSTTR** — Mean Segmental Type-Token Ratio; a length-unbiased vocabulary-richness
  measure computed by averaging type-token ratio over fixed-size windows. §1.10.
- **Signature term / cliché term** — a word the brand (or the generic corpus)
  over-uses relative to the other, by at least a 2× "lift" ratio. §1.10.
- **Flesch-Kincaid grade level** — a standard readability formula based on average
  sentence length and average syllables per word. §1.11.
- **Style envelope** — a brand's mean ± standard deviation across six structural
  features (sentence length, its variance, readability, bullet usage, passive-voice
  rate, exclamation rate), used to z-score new copy. §1.11.
- **LLM-as-judge** — using a language model to rate text on fixed, narrow, named axes
  (never to directly produce the final score), so the actual metric stays
  auditable/reproducible even though one input to it came from a model. §1.12.
- **Quadrant / 2×2** — the four-way classification (IDEAL / ON-BRAND BUT GENERIC /
  UNIQUE BUT OFF-BRAND / LOST) produced by thresholding consistency and
  distinctiveness independently, never blending them into one number. §1.13.
- **MVBF** — Minimum Viable Brand Fingerprint; the six required asset types (`name,
  tagline, mission, vision, values, about`) without which a brand can't be scored at
  all. §2.2.
- **Chunk / Fingerprint / GenericCorpus / InputCopy** — the core Pydantic data model;
  see Part 2.
- **Fit vs. encode** — fitting builds a vector space from a corpus (happens exactly
  once, offline, per brand); encoding projects new text through an already-fitted
  space (the only thing allowed at request time). §1.2, §3.1, §4.2.

---

# Appendix B — File Map

| File | One-line purpose |
|---|---|
| `loci/fingerprint.py` | Data model: `Layer`, `ASSET_LAYER_MAP`, `Chunk`, `BrandFingerprint`, MVBF/`is_scorable()`, `GenericCorpus`, `InputCopy`. |
| `loci/vectormath.py` | Shared, stateless vector math: `l2_normalise`, `centroid`, `cosine`. |
| `loci/input_encoder.py` | Loads a fitted vectorizer + term weights; encodes new text at request time. No `.fit()`. |
| `loci/vector_store.py` | `BrandVectorStore` — loads a brand's on-disk vectors/FAISS indices/encoder into memory; per-layer slicing and calibration-subset rules; FAISS nearest-neighbor lookups. |
| `loci/metrics/centroid.py` | `DualCentroidModel` — the dual-centroid, calibrated consistency/distinctiveness core. |
| `loci/metrics/lexical.py` | `LexicalScorer` — MSTTR diversity, signature-term hit rate, cliché density. |
| `loci/metrics/structural.py` | `StructuralScorer` — stylometric z-scoring against a brand's style envelope. |
| `loci/metrics/tone.py` | `ToneScorer` — LLM-as-judge tone axes, heuristic fallback, exact-text cache, daily call budget, usage tracker. |
| `loci/scorer.py` | `BrandDistinctivenessScorer` — the aggregator; `LAYER_WEIGHTS`, `_blend()`, quadrant lookup, overall aggregation, `Report`. |
| `vector_generation/embedder.py` | `TfidfEmbedder` (fit + discriminative reweighting), `SentenceTransformerEmbedder` (dense alternative). The only module allowed to `.fit()`. |
| `vector_generation/generate_embeddings.py` | `generate_from_dicts()` — the shared offline/live generation core; atomic-write pattern; CLI entry point. |
| `vector_generation/industries.py` | The immutable, append-only industry corpus registry. |
| `vector_generation/jobs.py` | In-memory job state machine for live (API-triggered) generation. |
| `api.py` | FastAPI app — every HTTP endpoint, rate limiting, CORS, concurrency guard, startup/shutdown lifecycle. |
| `demo.py` | CLI worked example — scores four candidate texts against the Duolingo fingerprint, one per quadrant. |
| `render.yaml` | Render Blueprint deployment config — single instance, single worker, by design. |
| `requirements.txt` | Exact-pinned dependencies for reproducible builds. |
| `.python-version` | Pins the Python interpreter version to match `render.yaml`. |
| `data/*.json` | Example brand fingerprints, generic corpora, and sample request payloads used by `demo.py` and for manual testing. |
