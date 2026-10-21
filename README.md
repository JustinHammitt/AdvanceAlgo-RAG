# Query-anchored expansion: example deliverable

**`main.py` is the main implementation. Start here and make future code changes here.**

Run the example:

```powershell
python main.py
python -m unittest discover -s tests -t . -v
```

`first_code_deliverable.ipynb` is retained only as reference material. It is not
required to run or test the project. The example is
"Which scientist developed the theory of general relativity?" with the existing
Einstein passage. The default command does not download SQuAD or run evaluation.

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

Results are written to `evaluation_results/example_results.csv` and
`evaluation_results/example_results.json`. CSV includes one row per comparison, all metrics,
prompts, responses, errors, and selected spans (nested data uses JSON cells). Empty
score cells indicate missing or failed responses. The JSON includes the fixed evidence
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

Fixed-Size and Adaptive-Chunking use the same ranking and selection.
Adaptive-Chunking evaluates recursive, paragraph, and sentence strategies using
the authors' five intrinsic metrics and chooses the highest-scoring strategy per
document. Install its additional NLP dependencies before running either evaluation:

```powershell
python -m pip install -r requirements-adaptive.txt
python -m spacy download en_core_web_sm
```

See [Adaptive Chunking setup and methodology](docs/adaptive_chunking.md) for the
models, candidate-pool adaptation, undefined metric handling, and audit output.
Preprocessing is reported separately from retrieval time. Timings are single-run
diagnostics. The default Einstein retrieval demo remains dependency-free.

All six proposed B/budget combinations and the existing 32/64 demonstration
are evaluated. The 46-token passage fits every context budget, so all methods
retrieve the full passage: this example validates execution, not comparative
performance or dataset-wide accuracy.

## SQuAD evaluation

The entry point remains `main.py`; `squad_evaluation.py` handles dataset loading
and the experiment. Existing commands keep their behavior:

| Command | Behavior |
|---|---|
| `python main.py` | Print context for the hardcoded Einstein question |
| `python main.py --evaluate` | Evaluate the Einstein example with Ollama |
| `python main.py --squad-evaluate` | Evaluate the 300-question SQuAD experiment with Ollama |

Configure `.env` first. The SQuAD command downloads and caches the official
[SQuAD 1.1 public development dataset](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
in `data/dev-v1.1.json`. The [dataset project](https://rajpurkar.github.io/SQuAD-explorer/)
distributes the data under CC BY-SA 4.0. Use `--squad-file path/to/file.json` to
use a local copy without downloading. SQuAD 2.0 is not supported.

The default seed is 42. Ten eligible articles are sampled, with thirty questions
per article: two articles supply 60 project development questions, and eight
separate articles supply 240 project test questions. These are custom splits
of the public data, not official SQuAD test results. Selection does not use model
results or keyword-match success. Use development results for tuning and keep
test results separate. Both splits run by default; to run only development:

```powershell
python main.py --squad-evaluate --squad-split dev
```

For a small trial (two questions, eighteen comparisons per question):

```powershell
python main.py --squad-evaluate --squad-split dev --squad-limit 2
```

`--squad-seed` changes the reproducible selection. Every full run compares all
three methods across chunk limits 128/256/512 and budgets 512/1024: 5,400
comparison rows. Identical prompts within a question reuse the same response;
different contexts can require separate model calls. The candidate collection
contains **all paragraphs of the question's article**, identically for every
method, rather than only the known answer paragraph. Article selection is
oracle-provided; this does not evaluate a corpus-wide first-stage retriever.

Results go in a fresh `evaluation_results/squad_<UTC timestamp>/` folder:

- `manifest.json`: dataset checksum, model/settings, seed, all 300 selected IDs,
  split assignments, and IDs actually evaluated.
- `questions.json`: selected questions and reference answers.
- `evidence_labels.json`: labels fixed before retrieval, in original character offsets.
- `results.csv` and `results.jsonl`: one flushed record per comparison with spans, prompt, response,
  answer/evidence metrics, latency, and generation errors.
- `summary.csv` and `summary.json`: averages separately by split,
  method, and configuration. CSV preserves full numeric precision and opens in
  Excel. CSV also includes retrieval
  latency, no-match rate, and generation success/error counts.
- `preprocessing.csv` and `preprocessing.json`: one-time baseline construction cost per article/chunk limit.
- `adaptive_selection.json`: candidate strategies, five metric scores, selected
  strategy, and model configuration for each input document/chunk limit.

Evaluation runs do not generate Markdown reports.
Completed rows survive interruption; JSON and CSV summaries update after each question.
A new invocation starts a new run, rather than resuming an interrupted one.
Generated results and cached data are Git-ignored. Preserve/share chosen results
explicitly when recording the final experiment. No endpoint is stored in the
manifest, though network error messages can include connection details.

To export an older run without calling Ollama again, use its saved `summary.json`:

```python
import json
from pathlib import Path
from squad_evaluation import write_summary_csv

run = Path('evaluation_results/squad_YOUR_RUN_FOLDER')
write_summary_csv(json.loads((run / 'summary.json').read_text(encoding='utf-8')),
                  run / 'summary.csv')
```

### Evidence labels and metric limits

SQuAD supplies answer spans, not independent complete supporting-evidence labels.
By default, evidence metrics use a clearly marked `answer_sentence_proxy`:
sentences overlapping reference answer spans, determined before retrieval.
Sentence splitting is a punctuation heuristic. Multiple annotated answers use
the union of those sentences for evidence metrics; answer EM/F1 instead takes
the best match over valid references. Proxy evidence metrics are exploratory,
not manually validated evidence completeness.

For final evidence metrics, supply reviewed labels with
`--evidence-labels reviewed_labels.json`. The file maps every evaluated question
ID to a nonempty list of intervals; offsets refer to the original paragraph,
are zero-based, and exclude `end_char`:

```json
{
  "question-id": [
    {"document_id": 0, "start_char": 0, "end_char": 100}
  ]
}
```

`document_id` is the paragraph's zero-based index within the article. All labels
are validated before generation. The raw dataset is sufficient to establish
these labels independently of retrieved results; do not tune labels to outputs.
The exported proxy-label file uses the same schema for review.

No-match questions remain in the run. Summaries include the no-match rate,
evidence recall/precision/completeness, context size, retrieval latency, and
generation success/error counts. EM/F1 are fractions from 0 to 1. Failed model
calls have null raw scores and count as zero in aggregate accuracy, so failures
cannot inflate scores by dropping questions. Baseline preprocessing is cached
and reported separately; query-anchored construction is timed at query time.
Word-token budgets have the same limitations described above. Adaptive Chunking
uses the paper's metric implementation with our SQuAD candidate pool; it is not
an exact reproduction of the paper's PDF experiment. Old Sentence-Adaptive
results measured the former sentence-packing baseline and should not be relabeled
or combined with new Adaptive-Chunking results.
