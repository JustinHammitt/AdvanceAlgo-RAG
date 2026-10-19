# Query-anchored expansion: example deliverable

**`main.py` is the main implementation. Start here and make future code changes here.**

Run the example:

```powershell
python main.py
python -m unittest test_first_code_deliverable -v
```

`first_code_deliverable.ipynb` is retained only as reference material. It is not
required to run or test the project. The example is
"Which scientist developed the theory of general relativity?" with the existing
Einstein passage. No SQuAD download or full dataset experiment is performed.

The main algorithm is `query_anchored_expansion()` in `main.py`: it extracts
query terms, locates anchors, expands and merges regions, ranks them, and selects
context within the token budget. Running `python main.py` prints the example's
query terms and selected spans without calling Ollama or running evaluation.

Evaluation runs only when explicitly requested:

```powershell
python main.py --evaluate
```

There is no default endpoint or model. Copy the configuration template:

```powershell
Copy-Item example.env .env
```

Fill in `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in `.env`, then run
`python main.py --evaluate`. `.env` is already ignored by Git. The file uses
one `KEY=value` per line, with optional matching quotes and full-line comments.
Shell environment variables override `.env`; explicit Python arguments override
both. Evaluation reports a configuration error if either value is missing.
The ordinary example does not require configuration or contact Ollama.

You can also call evaluation from Python using the same configuration:

```python
from main import evaluate_example

example_report = evaluate_example()
```

Results are written to `evaluation_results/example_results.md` and
`evaluation_results/example_results.json`. The JSON includes the fixed evidence
labels, selected spans, prompts, actual model responses, errors, and settings.
Missing or failed model responses yield pending/error status and null EM/F1,
never fabricated scores. Identical prompts reuse the same response within a run.

Keyword extraction lowercases, removes common function/question words, and
deduplicates terms. Exact matching does not resolve synonyms or inflections.
Merging requires touching/overlapping spans, a combined length within B, and
either a newly represented term or increased relevance relative to the current
span. Relevance uses 0.7 distinct-term coverage + 0.3 matched-occurrence density;
the same score and configurable weights are used for ranking.

Evidence recall and precision compare unique retrieved source-token positions
with independently specified evidence positions. Completeness requires all
labeled evidence tokens. Answer EM/F1 use normalized answer text, separately
from evidence metrics. Context size uses regex word tokens, not LLM tokens.

Fixed-size and sentence-adaptive baselines use the same ranking and selection.
Sentence-adaptive greedily packs sentences within B and splits oversized
sentences; it is a simplified structural baseline, not a reproduction of the
published Adaptive Chunking strategy-selection algorithm. Preprocessing is
reported separately from retrieval time. Timings are single-run diagnostics.

All six proposed B/budget combinations and the existing 32/64 demonstration
are evaluated. The 46-token passage fits every context budget, so all methods
retrieve the full passage: this example validates execution, not comparative
performance or dataset-wide accuracy.
