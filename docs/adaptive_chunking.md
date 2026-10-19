# Adaptive Chunking baseline

`adaptive_baseline.py` selects a chunking strategy independently for each input
document before query ranking. `main.py` and `--squad-evaluate` now report this
method as **Adaptive-Chunking**, replacing **Sentence-Adaptive**. Old result files
are not converted or relabeled: they measured a different baseline.

## Source and scope

The method is from de Moura Junior, Lelong, and Blangero,
[Adaptive Chunking: Optimizing Chunking-Method Selection for RAG](https://arxiv.org/abs/2603.25333),
LREC 2026. We call the metric functions and recursive splitter in the
[authors' MIT-licensed implementation](https://github.com/ekimetrics/adaptive-chunking),
pinned to commit `ea87ce8e1a97888f3f179e7f1359ff7f43fb179d`.

This is the five-metric selection method adapted to our SQuAD experiment, not a
reproduction of the paper's PDF dataset, GPT-generated regex candidate, or exact
candidate pool. The paper's page splitter has no native equivalent in SQuAD.
The project keeps its word-token budgets and common ranking rule.

## Setup

Use Python 3.11 or later in your project environment:

```powershell
python -m pip install -r requirements-adaptive.txt
python -m spacy download en_core_web_sm
```

The first adaptive run loads/downloads Jina embeddings and Maverick coreference
models from Hugging Face. This requires network access, disk space, and substantially
more RAM and preprocessing time than the former sentence-packing method. CPU is
the default; set `ADAPTIVE_DEVICE=cuda:0` in the shell when supported by your PyTorch
installation. These NLP models run in the Python process; Ollama still handles
answer generation using `.env` configuration. The embedding implementation loads
the model's custom Python code (`trust_remote_code=True`), as required by Jina v3.

Optional shell environment variables (not read from `.env`):

- `ADAPTIVE_DEVICE`: default `cpu`.
- `ADAPTIVE_EMBEDDING_MODEL`: default `jinaai/jina-embeddings-v3`.
- `ADAPTIVE_COREF_MODEL`: default `sapienzanlp/maverick-mes-ontonotes`.

Changing these models changes the experiment; actual names are recorded in
reports. Maverick's optional upstream dependency is CC BY-NC-SA 4.0; see the
authors' repository notices and model cards. The ordinary `python main.py`
Einstein retrieval demo requires none of these dependencies and does not load models.

## Candidate construction and selection

For every normalized input document and maximum size `B`, evaluate:

1. The authors' recursive split-then-merge strategy with target `B`.
2. The same strategy with target `max(1, B // 2)`.
3. Paragraph boundaries, with oversized blocks split and small neighbors merged.
4. Sentence boundaries from spaCy, with the same size postprocessing.

The minimum desirable size is `min(100, B)`. All candidate regions are mapped to
our tokenizer and constrained to `B`. Their source-token coverage must be complete
and non-overlapping. Character slices retain whitespace and punctuation for
metric computation. Zero-token documents produce no spans. Normalization is the
same lowercasing used by the other methods. In SQuAD each paragraph is currently
an input document, so paragraph integrity can have limited discriminatory power.

The five scores are computed by the upstream implementation:

| Metric | Computation |
|---|---|
| RC | One minus the fraction of Maverick-derived entity/pronoun pairs broken by chunk boundaries |
| ICC | Semantic similarity of sentence pieces to their chunk embedding |
| DCC | Semantic similarity of chunks to their surrounding windows, up to 3,000 word tokens |
| BI | Fraction of paragraph blocks kept intact, with the upstream five-character tolerance |
| SC | Fraction of chunks within the minimum and maximum sizes |

Choose the candidate with the highest unweighted mean of available metric
scores. A metric can be undefined: no entity/pronoun pairs for RC, no multi-sentence
chunks for ICC, or no multi-chunk windows for DCC. These remain `null`, are excluded
from the mean, and are explicitly listed in the audit. They are not replaced by
fabricated perfect scores. Candidate order breaks ties deterministically. No
question, reference answer, or evidence label is passed to the selector.

The selected spans then use exactly the same query ranking and context-budget
selection as Fixed-Size and Query-Anchored. Models initialize once, document
analysis is cached across chunk limits, and selected candidate regions are cached
across questions/budgets. Initialization time is reported separately from
document preprocessing and query-time ranking.

## Audit outputs

SQuAD writes `adaptive_selection.json`: each document and chunk limit records
every candidate's token intervals, RC/ICC/DCC/BI/SC, available metrics, aggregate
score, chosen strategy, and model configuration. The manifest records backend
configuration and initialization time. The Einstein JSON includes the same audit.
`preprocessing.json` includes candidate generation and scoring time. Existing
summary CSV, Markdown, and JSON outputs retain their columns and use the new
method name.

## Validation scope

Offline tests use an explicitly synthetic backend to verify strategy selection,
tie handling, missing metrics, invalid scores, complete coverage, hard limits,
offsets, caching, public dispatch, and the evaluation/audit exports. Those fixture
scores are never used by production code. Missing dependencies or failed model
loads stop the evaluation; there is no automatic heuristic fallback. Real-model
evaluation must be run after installing the NLP dependencies and models.
