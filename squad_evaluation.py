"""SQuAD 1.1 experiment runner. Invoke through main.py --squad-evaluate."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from time import perf_counter
import urllib.request

import main as core

DATA_URL = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'
METHODS = ('Query-Anchored', 'Fixed-Size', 'Sentence-Adaptive')
CONFIGURATIONS = [(b, budget) for b in (128, 256, 512) for budget in (512, 1024)]


def load_dataset(path=None):
    """Download only the default cache; an explicit missing path is an error."""
    location = Path(path) if path else Path(__file__).parent / 'data' / 'dev-v1.1.json'
    if not location.exists() and path is None:
        print('Downloading SQuAD 1.1 from Stanford...', flush=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
        payload = json.loads(content)
        if payload.get('version') != '1.1':
            raise ValueError('Expected SQuAD version 1.1')
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_bytes(content)
    content = location.read_bytes()
    payload = json.loads(content)
    if payload.get('version') != '1.1' or not isinstance(payload.get('data'), list):
        raise ValueError('Expected a SQuAD 1.1 JSON file with a data array')
    return payload['data'], hashlib.sha256(content).hexdigest(), location


def select_questions(articles, seed=42):
    """Thirty questions/article: two dev articles and eight test articles."""
    eligible, seen_ids, seen_titles = [], set(), set()
    for article in articles:
        title = article['title']
        if title in seen_titles:
            raise ValueError('Duplicate article title: ' + title)
        seen_titles.add(title)
        documents = [p['context'] for p in article['paragraphs']]
        questions = []
        for doc_id, paragraph in enumerate(article['paragraphs']):
            for qa in paragraph['qas']:
                if qa['id'] in seen_ids:
                    raise ValueError('Duplicate question ID: ' + qa['id'])
                seen_ids.add(qa['id'])
                if not qa.get('answers') or qa.get('is_impossible'):
                    raise ValueError('SQuAD 1.1 requires answerable questions: ' + qa['id'])
                for answer in qa['answers']:
                    start, text = answer['answer_start'], answer['text']
                    if start < 0 or not text or paragraph['context'][start:start+len(text)] != text:
                        raise ValueError('Invalid answer offset: ' + qa['id'])
                questions.append(dict(id=qa['id'], question=qa['question'],
                                      answers=qa['answers'], document_id=doc_id,
                                      title=title, documents=documents))
        if len(questions) >= 30:
            eligible.append((title, sorted(questions, key=lambda q: q['id'])))
    if len(eligible) < 10:
        raise ValueError('Need at least ten articles with thirty questions each for the proposal split')
    rng = random.Random(seed)
    chosen = rng.sample(sorted(eligible), 10)
    selected = []
    for index, (_, questions) in enumerate(chosen):
        for question in rng.sample(questions, 30):
            selected.append({**question, 'split': 'dev' if index < 2 else 'test'})
    return selected


def sentence_evidence(question):
    """Proxy labels fixed before retrieval: sentences intersecting answer spans."""
    doc_id = question['document_id']
    source = question['documents'][doc_id]
    boundaries = [0] + [m.end() for m in re.finditer(r'[.!?](?:["\u201d\u2019])?\s+', source)] + [len(source)]
    intervals = set()
    for answer in question['answers']:
        start, end = answer['answer_start'], answer['answer_start'] + len(answer['text'])
        for left, right in zip(boundaries, boundaries[1:]):
            if left < end and right > start:
                intervals.add((left, right))
    return [dict(document_id=doc_id, start_char=a, end_char=b) for a, b in sorted(intervals)]


def token_evidence(question, labels):
    """Convert original character labels to normalized word-token coordinates."""
    evidence = []
    for label in labels:
        doc_id, a, b = (label[k] for k in ('document_id', 'start_char', 'end_char'))
        if not all(isinstance(v, int) for v in (doc_id, a, b)) or not 0 <= doc_id < len(question['documents']):
            raise ValueError('Invalid evidence document/offset for ' + question['id'])
        original = question['documents'][doc_id]
        if not 0 <= a < b <= len(original):
            raise ValueError('Evidence offsets out of range for ' + question['id'])
        # Lowercasing can expand Unicode characters; never reuse original offsets blindly.
        start, end = len(original[:a].lower()), len(original[:b].lower())
        tokens = core.tokenize_document(original.lower())
        covered = [i for i, (_, left, right) in enumerate(tokens) if left < end and right > start]
        if not covered:
            raise ValueError('Evidence label contains no tokens for ' + question['id'])
        evidence.append(dict(document_id=doc_id, start_token=min(covered), end_token=max(covered)+1))
    if not evidence:
        raise ValueError('Missing evidence labels for ' + question['id'])
    return evidence


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row['method'], row['chunk_limit'], row['context_budget'])].append(row)
    result = []
    for (split, method, B, budget), group in sorted(groups.items()):
        successful = [r for r in group if r['generation_status'] == 'complete']
        summary = dict(split=split, method=method, chunk_limit=B, context_budget=budget,
                       questions=len(group), generation_successes=len(successful),
                       generation_errors=len(group)-len(successful))
        for metric in ('evidence_recall', 'precision', 'completeness', 'context_tokens',
                       'retrieval_ms', 'no_query_match'):
            summary[metric] = sum(r[metric] for r in group) / len(group)
        # Failed generations stay in the denominator instead of inflating accuracy.
        for metric in ('exact_match', 'answer_f1'):
            summary[metric] = sum(r[metric] or 0 for r in group) / len(group)
        result.append(summary)
    return result


def evaluate_squad(dataset_path=None, seed=42, split='all', limit=None,
                   evidence_path=None, output_root='evaluation_results'):
    if split not in ('all', 'dev', 'test') or (limit is not None and limit < 1):
        raise ValueError('Choose dev/test/all and a positive question limit')
    base_url, model = core.ollama_config()
    articles, checksum, location = load_dataset(dataset_path)
    selection = select_questions(articles, seed)
    questions = [q for q in selection if split == 'all' or q['split'] == split]
    if limit is not None:
        questions = questions[:limit]
    reviewed = json.loads(Path(evidence_path).read_text(encoding='utf-8')) if evidence_path else None
    labels = {}
    for q in questions:
        if reviewed is not None and q['id'] not in reviewed:
            raise ValueError('Reviewed evidence missing question ' + q['id'])
        labels[q['id']] = reviewed[q['id']] if reviewed is not None else sentence_evidence(q)
        token_evidence(q, labels[q['id']])  # Validate every label before any model call.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(output_root) / ('squad_' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(dataset=str(location.resolve()), dataset_sha256=checksum, source_url=DATA_URL,
                    seed=seed, model=model, split=split, limit=limit,
                    generation_options=dict(temperature=0, seed=42, num_predict=64, num_ctx=4096),
                    configurations=CONFIGURATIONS, methods=METHODS,
                    evidence_kind='reviewed' if reviewed is not None else 'answer_sentence_proxy',
                    candidate_scope='all paragraphs of the question article; oracle article selection',
                    selection=[{k: q[k] for k in ('id', 'title', 'split', 'document_id')} for q in selection],
                    run_question_ids=[q['id'] for q in questions],
                    note='Project dev/test splits of SQuAD public dev data, not official SQuAD test scores. Sentence-Adaptive is a simplified baseline.')
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    (output / 'evidence_labels.json').write_text(json.dumps(labels, indent=2)+'\n', encoding='utf-8')
    snapshots = [{k: q[k] for k in ('id', 'title', 'split', 'question', 'answers', 'document_id')} for q in questions]
    (output / 'questions.json').write_text(json.dumps(snapshots, indent=2)+'\n', encoding='utf-8')
    print(f'Running {len(questions)} questions, {len(CONFIGURATIONS)*len(METHODS)} comparisons each. Results: {output}', flush=True)
    rows, prepared, preparation = [], {}, []
    with (output / 'results.jsonl').open('w', encoding='utf-8') as log:
        for number, q in enumerate(questions, 1):
            terms = core.extract_query_terms_from_query(q['question'])
            documents = q['documents']
            no_match = not any(t[0] in terms for d in documents for t in core.tokenize_document(d.lower()))
            evidence = token_evidence(q, labels[q['id']])
            answer_cache = {}
            for B, budget in CONFIGURATIONS:
                for method in METHODS:
                    if method != 'Query-Anchored':
                        key = (q['title'], B, method)
                        if key not in prepared:
                            start = perf_counter()
                            prepared[key] = core.baseline_documents(documents, B, method == 'Sentence-Adaptive')
                            preparation.append(dict(title=q['title'], chunk_limit=B, method=method,
                                                    preprocessing_ms=(perf_counter()-start)*1000))
                    start = perf_counter()
                    if method == 'Query-Anchored':
                        spans = core.query_anchored_expansion(q['question'], documents, B, budget)
                    else:
                        candidates = prepared[key]
                        ranked = core.rank_candidate_spans(candidates, terms)
                        spans = core.select_spans_within_context_budget(ranked, candidates, terms, budget)
                    latency = (perf_counter()-start)*1000
                    context = '\n'.join(s['text'] for s in sorted(spans, key=lambda s: (s['document_id'], s['start_token'])))
                    prompt = ('Answer the question using only the context. Return only the short answer, '
                              'without explanation. If the context does not answer it, return UNKNOWN.\n\n'
                              f'Context:\n{context}\n\nQuestion: {q["question"]}\nAnswer:')
                    prediction, error = None, None
                    cache_hit = prompt in answer_cache
                    started = perf_counter()
                    try:
                        if not cache_hit:
                            answer_cache[prompt] = core.ollama_answer(prompt, base_url, model)
                        prediction = answer_cache[prompt]
                    except Exception as exc:
                        error = str(exc)
                    metrics = core.answer_scores(prediction, [a['text'] for a in q['answers']]) if prediction is not None else dict(exact_match=None, answer_f1=None)
                    row = dict(question_id=q['id'], title=q['title'], split=q['split'],
                               method=method, chunk_limit=B, context_budget=budget,
                               **core.evidence_scores(spans, evidence), **metrics,
                               no_query_match=no_match, retrieval_ms=latency,
                               generation_ms=(perf_counter()-started)*1000, response_reused=cache_hit,
                               prediction=prediction, generation_error=error,
                               generation_status='complete' if prediction is not None else 'error',
                               selected_spans=spans, prompt=prompt)
                    rows.append(row)
                    log.write(json.dumps(row)+'\n')
                    log.flush()
            print(f'[{number}/{len(questions)}] {q["split"]}: {q["id"]}', flush=True)
            # Save summaries after each question, so interruptions retain completed work.
            (output / 'summary.json').write_text(json.dumps(summarize(rows), indent=2)+'\n', encoding='utf-8')
            (output / 'preprocessing.json').write_text(json.dumps(preparation, indent=2)+'\n', encoding='utf-8')
    summary = summarize(rows)
    lines = ['# SQuAD evaluation', '', metadata['note'], '',
             'Evidence labels: ' + metadata['evidence_kind'],
             'Failed generations count as zero in aggregate EM/F1; raw scores are null.', '',
             '| Split | Method | B | Budget | N | Recall | Precision | Complete | EM | F1 | Tokens | Errors |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        values = [r[k] for k in ('split', 'method', 'chunk_limit', 'context_budget', 'questions',
                                'evidence_recall', 'precision', 'completeness', 'exact_match',
                                'answer_f1', 'context_tokens', 'generation_errors')]
        lines.append('| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in values) + ' |')
    (output / 'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Finished. Read {output / "summary.md"}', flush=True)
    return output
