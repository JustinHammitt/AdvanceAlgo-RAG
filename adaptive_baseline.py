"""Document-level Adaptive Chunking using the authors' intrinsic metrics.

The candidate pool is adapted to plain-text SQuAD and our hard word-token cap.
See docs/adaptive_chunking.md for the paper, configuration, and differences.
"""
from functools import lru_cache
import math
import os
import re

UPSTREAM_COMMIT = 'ea87ce8e1a97888f3f179e7f1359ff7f43fb179d'
METRICS = ('RC', 'ICC', 'DCC', 'BI', 'SC')


def tokens_of(text):
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r'\b\w+\b', text)]


def regions_from_chunks(text, tokens, chunks, B):
    """Map lossless character chunks to a complete, non-overlapping token partition."""
    if ''.join(chunks) != text:
        raise ValueError('Adaptive candidate must preserve the complete source text')
    cuts, cursor = {0, len(tokens)}, 0
    for chunk in chunks[:-1]:
        cursor += len(chunk)
        # Snap a character split inside a word to that word's start.
        cuts.add(next((i for i, (_, a, b) in enumerate(tokens) if b > cursor), len(tokens)))
    cuts = sorted(cuts)
    regions = []
    for a, b in zip(cuts, cuts[1:]):
        regions.extend((start, min(start+B, b)) for start in range(a, b, B))
    return regions


def chunks_from_regions(text, tokens, regions):
    """Include separators once, preserving source text for upstream offset metrics."""
    if not regions:
        return []
    edges = [0] + [tokens[a][1] for a, _ in regions[1:]] + [len(text)]
    return [text[a:b] for a, b in zip(edges, edges[1:])]


def merge_small_regions(regions, minimum, B):
    """Postprocess small units only when the hard cap can still be respected."""
    output = []
    for a, b in regions:
        if output and output[-1][1]-output[-1][0] < minimum and b-output[-1][0] <= B:
            output[-1] = (output[-1][0], b)
        else:
            output.append((a, b))
    if len(output) > 1 and output[-1][1]-output[-1][0] < minimum:
        if output[-1][1]-output[-2][0] <= B:
            output[-2:] = [(output[-2][0], output[-1][1])]
    return output


class _CorefBlocks:
    def split_text(self, text):
        # Lossless small blocks keep upstream coreference windows from cropping
        # long paragraphs. Adjacent blocks are concatenated before inference.
        return [text[i:i+128] for i in range(0, len(text), 128)]


class _CachedEmbedder:
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def encode(self, texts, **kwargs):
        import numpy as np
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            embeddings = self.model.encode(missing, **kwargs)
            self.cache.update(zip(missing, embeddings))
        return np.asarray([self.cache[t] for t in texts])


class PaperBackend:
    """Real embeddings, Maverick coreference, and upstream scoring functions.

    Missing dependencies/models are errors; never substitute lexical heuristics.
    Imports and model downloads happen only when Adaptive Chunking is requested.
    """
    def __init__(self):
        try:
            import spacy
            from sentence_transformers import SentenceTransformer
            from adaptive_chunking import metrics
            from adaptive_chunking.splitters import RecursiveSplitter
        except ImportError as exc:
            raise ValueError('Adaptive Chunking requires Python 3.11+ and its NLP dependencies. '
                             'Run: python -m pip install -r requirements-adaptive.txt; '
                             'then python -m spacy download en_core_web_sm') from exc
        self.metrics = metrics
        self.splitter = RecursiveSplitter
        device = os.environ.get('ADAPTIVE_DEVICE', 'cpu')
        self.embedding_name = os.environ.get('ADAPTIVE_EMBEDDING_MODEL', 'jinaai/jina-embeddings-v3')
        self.coref_name = os.environ.get('ADAPTIVE_COREF_MODEL', 'sapienzanlp/maverick-mes-ontonotes')
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.embedder = _CachedEmbedder(SentenceTransformer(
                self.embedding_name, device=device, trust_remote_code=True))
            self.coref = metrics.CoreferenceSolver(
                pre_splitter=_CorefBlocks(), device=device, model_name=self.coref_name)
        except (ImportError, OSError) as exc:
            raise ValueError('Could not load Adaptive Chunking models. Follow docs/adaptive_chunking.md. '
                             + str(exc)) from exc
        self.device = device

    def describe(self):
        from importlib.metadata import version
        return dict(upstream_commit=UPSTREAM_COMMIT, embedding_model=self.embedding_name,
                    coreference_model=self.coref_name, device=self.device,
                    package_version=version('adaptive-chunking'), sentence_model='en_core_web_sm',
                    token_unit='regex_word', window_context_tokens=3000, block_tolerance_chars=5)

    def analyze(self, text):
        self.embedder.cache.clear()  # bounded to the most recently scored document
        sentence_starts = [s.start_char for s in self.nlp(text).sents]
        block_starts = [m.end() for m in re.finditer(r'\n\s*\n', text) if m.end() < len(text)]
        mentions = self.coref.find_mentions(text)
        pairs = self.metrics.extract_entity_pronoun_pairs(text, mentions, self.nlp)
        return dict(sentence_starts=sentence_starts, block_starts=block_starts, pairs=pairs)

    def split(self, text, target, minimum):
        splitter = self.splitter(chunk_size=target, chunk_overlap=0,
                                 length_function=lambda s: len(tokens_of(s)),
                                 separators=['\n\n', '\n', '. ', '? ', '! ', '; ', ', ', ' ', ''],
                                 attach_separator_to='end', merging='small_only',
                                 min_chunk_tokens=min(minimum, target), max_tokens_strategy='chunk_size')
        return splitter.split_text(text)

    def score(self, text, chunks, analysis, minimum, B):
        m = self.metrics
        embeddings = m.compute_chunk_embeddings(chunks, self.embedder)
        error = (m.compute_filtered_missing_ref_error(text, chunks, analysis['pairs'])
                 if analysis['pairs'] else None)
        values = dict(
            RC=None if error is None else 1-error,
            ICC=m.compute_intrachunk_cohesion(chunks, text, analysis['sentence_starts'],
                                              self.embedder, chunk_embeddings=embeddings),
            DCC=m.compute_contextual_coherence(chunks, text, self.embedder,
                                               count_tokens_func=lambda s: len(tokens_of(s)),
                                               window_context_tokens=3000, chunk_embeddings=embeddings),
            BI=m.compute_block_integrity(chunks, analysis['block_starts'], text, tolerance_chars=5),
            SC=m.compute_size_compliance(chunks, max_tokens=B, min_tokens=minimum,
                                         count_tokens_func=lambda s: len(tokens_of(s))))
        return {k: None if v is None else float(v) for k, v in values.items()}


class AdaptiveChunker:
    def __init__(self, backend=None):
        self.backend = backend if backend is not None else PaperBackend()
        self.analysis_cache = {}
        self.result_cache = {}

    def build_documents(self, documents, B):
        if B < 1:
            raise ValueError('B must be positive')
        return [dict(self._build(text.lower(), B), document_id=i) for i, text in enumerate(documents)]

    def _build(self, text, B):
        key = (text, B)
        if key in self.result_cache:
            return self.result_cache[key]
        tokens = tokens_of(text)
        record = dict(document_text=text, tokens=tokens, anchor_positions={}, regions=[])
        if not tokens:
            record['adaptive_selection'] = dict(strategy='empty', candidates=[], backend=self.backend.describe())
            return record
        if text not in self.analysis_cache:
            self.analysis_cache[text] = self.backend.analyze(text)
        analysis = self.analysis_cache[text]
        minimum = min(100, B)
        pool = []
        for name, target in [('recursive_full', B), ('recursive_half', max(1, B//2))]:
            chunks = self.backend.split(text, target, minimum)
            pool.append((name, regions_from_chunks(text, tokens, chunks, B)))
        for name, starts in [('paragraph', analysis['block_starts']), ('sentence', analysis['sentence_starts'])]:
            points = sorted({0, len(text), *starts})
            chunks = [text[a:b] for a, b in zip(points, points[1:])]
            pool.append((name, merge_small_regions(regions_from_chunks(text, tokens, chunks, B), minimum, B)))
        scored = []
        metric_cache = {}
        for name, regions in pool:
            signature = tuple(regions)
            if signature not in metric_cache:
                values = self.backend.score(text, chunks_from_regions(text, tokens, regions), analysis, minimum, B)
                if set(values) != set(METRICS):
                    raise ValueError('Adaptive backend must return all five metric names')
                if any(v is not None and (not math.isfinite(v) or not 0 <= v <= 1) for v in values.values()):
                    raise ValueError('Adaptive metrics must be finite scores in [0, 1] or None')
                metric_cache[signature] = values
            values = metric_cache[signature]
            available = [v for v in values.values() if v is not None]
            if not available:
                raise ValueError('No applicable Adaptive Chunking metrics')
            scored.append(dict(strategy=name, regions=regions, metrics=values,
                               applicable_metrics=[k for k in METRICS if values[k] is not None],
                               mean_score=sum(available)/len(available)))
        # Stable candidate order resolves equal scores reproducibly.
        winner = max(scored, key=lambda item: item['mean_score'])
        record['regions'] = winner['regions']
        record['adaptive_selection'] = dict(strategy=winner['strategy'], mean_score=winner['mean_score'],
                                             minimum_tokens=minimum, maximum_tokens=B,
                                             candidates=scored, backend=self.backend.describe())
        self.result_cache[key] = record
        return record


@lru_cache(maxsize=1)
def get_adaptive_chunker():
    return AdaptiveChunker()
