"""MAIN IMPLEMENTATION: query-anchored context expansion.

Run the example: python main.py
Run and record evaluation (including Ollama): python main.py --evaluate
The Jupyter notebook is reference material only; maintain this file.
"""

import string
import re
from collections import defaultdict
from collections import Counter
from pathlib import Path
from time import perf_counter
import json
import os
import urllib.request

EXAMPLE_QUERY = "Which scientist developed the theory of general relativity?"

EXAMPLE_DOCUMENT = """Albert Einstein was a German-born theoretical physicist. He developed the theory of special relativity in 1905. His work transformed modern physics. In 1915, Einstein completed the theory of general relativity, which describes gravity as the curvature of spacetime. His research also contributed to quantum mechanics."""

QUERY_STOP_WORDS = frozenset("""a an the and or but if then of in on at to for
from by with as is are was were be been being do does did have has had
which what who whom whose when where why how this that these those it its
he she they them their his her you your we our i me my s t""".split())

def query_anchored_expansion(
    query: str,
    documents: list[str],
    max_chunk_tokens: int = 128,
    context_budget_tokens: int = 512,
    CS_Weight: float = 0.7,
    DS_Weight: float = 0.3
) -> list[dict]:
    """
    Input -> Output Function. Output will then be used in evaluation phase.
    """

    ### 1. Clean input documents and query
    documents = clean_all_documents(documents)
    query = clean_query_text(query)

    ### 2. Extract query terms
    query_terms = extract_query_terms_from_query(query)
    query_terms = clean_query_terms(query_terms)

    ### 3. Identify anchors
    documents_and_anchor_positions = (
        get_anchor_positions_for_all_documents(
            documents=documents,
            query_terms=query_terms
        )
    )

    ### 4. Tokenize documents
    for document in documents_and_anchor_positions:
        document["tokens"] = tokenize_document(
            document["document_text"]
        )

    ### 5. Expand around anchors
    expanded_documents = expand_around_anchors(
        documents_and_anchor_positions=documents_and_anchor_positions,
        B=max_chunk_tokens
    )

    ### 6. Merge neighboring regions
    merged_documents = merge_anchoring_regions(
        expanded_documents=expanded_documents,
        documents_and_anchor_positions=documents_and_anchor_positions,
        query_terms=query_terms,
        CS_Weight=CS_Weight, DS_Weight=DS_Weight,
        B=max_chunk_tokens
    )

    ### 7. Rank candidate spans
    ranked_documents = rank_candidate_spans(
        merged_documents=merged_documents,
        query_terms=query_terms,
        CS_Weight=CS_Weight,
        DS_Weight=DS_Weight
    )

    ### 8. Select spans within context budget
    selected_spans = select_spans_within_context_budget(
        ranked_documents=ranked_documents,
        merged_documents=merged_documents,
        query_terms=query_terms,
        context_budget_tokens=context_budget_tokens
    )

    return selected_spans

def clean_document_text(document_text: str) -> str:
    """
    Returns the input document text with 
        1. all text converted to lower-case 
        2+. ???
    """
    ### Convert document text to lower-case
    document_text = document_text.lower()

    return document_text

def clean_query_text(query: str) -> str:
    """
    Returns the input query text with 
        1. all text converted to lower-case 
        2+. ???
    """
    ### Convert text to lower-case
    query = query.lower()

    return query

def clean_all_documents(documents: list[str]) -> list[str]:
    """
    Cleans all documents using clean_document_text. 
    This should work as-is once clean_document_text is finished.

    Returns list identical to input but with documents "cleaned".
    """
    result_list: list[str] = []

    for document in documents:
        cleaned_document = clean_document_text(document_text = document)
        result_list.append(cleaned_document)

    return result_list

def extract_query_terms_from_query(query: str) -> list[str]:
    return list(dict.fromkeys(
        word for word in re.findall(r"\b\w+\b", query.lower())
        if word not in QUERY_STOP_WORDS
    ))

def clean_query_terms(query_terms: list[str]) -> list[str]:
    """
    An extra just-in-case cleaning function.

    Should return a list identical to the input but with each term "cleaned" for input into the algorithm
    """
    query_terms = [term.lower() for term in query_terms]

    return query_terms

def get_anchor_positions_of_query_terms(
    query_terms: list[str],
    document_text: str,
    document_id: int
) -> dict[str, str | int | dict[str, list[int]]]:
    

    """
    Gets dictionary of lists containing the starting indices
    of each occurrence of a query term.

    Example return:
    {
        "document_text: "Einstein ................."
        "document_id": 1,
        "anchor_positions": {
            "developed": [60],
            "theory": [74, 181]
        }
    }

    Let
        * n = length of input_text in characters
        * k = number of query terms
        * q = total number of characters across all query terms
        * m = total number of matches found

    Time complexity: O(nk + q + m)
    """

    ### Combine query terms into one pattern
    pattern = r"\b(?:" + "|".join(map(re.escape, query_terms)) + r")\b"

    ### List out all patterns
    matches = defaultdict(list)

    for match in (re.finditer(pattern, document_text) if query_terms else []):
        matches[match.group()].append(match.start())

    return {
        "document_text": document_text, 
        "document_id": document_id,
        "anchor_positions": dict(matches)
    }

def get_anchor_positions_for_all_documents(documents: list[str],
                                           query_terms: list[str]
                                           ) -> list[dict[str, str | int | dict[str, list[int]]]]:
    """
    The document ID will be its 0-based index

    Let
        * n_i = length of input_text in characters in document i
        * k = number of query terms
        * q = total number of characters across all query terms
        * m_i = total number of matches found in document i
        * d = number of documents

    Time complexity: O(d * (nk + q + m))
    """
    all_word_anchor_positions: list[dict[str, int | dict[str, list[int]]]] = []

    for i in range(len(documents)):
        document_text = documents[i]
        word_anchor_positions = get_anchor_positions_of_query_terms(document_text=document_text,
                                                                    query_terms=query_terms,
                                                                    document_id=i)

        all_word_anchor_positions.append(word_anchor_positions)

    return all_word_anchor_positions

def tokenize_document(document_text: str) -> list[tuple[str, int, int]]:
    """
    Tokenizes document_text into individual words.

    Returns a list containing:
        (word, start_character, end_character)

    Time complexity: O(n)
    """

    tokens = [
        (match.group(), match.start(), match.end())
        for match in re.finditer(r"\b\w+\b", document_text)
    ]

    return tokens

def expand_around_anchors(
    documents_and_anchor_positions: list[dict],
    B: int
) -> list[dict]:
    """
    Expands regions around query-term anchors.

    Steps:
        1. Sort anchors by token position.
        2. Calculate midpoint boundaries between anchors.
        3. Expand each anchor up to B tokens, favoring balance.

    B = Maximum number of tokens per expanded region.

    Output should have 1 dictionary per document, in the format of
    [
        {
            "document_id": 0,
            "document_text": "...",
            "tokens": [
                ("albert", 0, 6),
                ("einstein", 7, 15),
                ...
            ],
            "anchor_positions": {
                "developed": [60],
                "theory": [74, 181]
            },
            "regions": [
                (5, 17),
                (23, 39)
            ]
        },
        {
            ...
        }
    ]
    """

    if B < 1:
        raise ValueError("B must be positive")

    results = []

    for document in documents_and_anchor_positions:

        ### Extract document information
        document_id = document["document_id"]
        document_text = document["document_text"]
        tokens = document["tokens"]
        anchor_positions = document["anchor_positions"]

        ### 1. Sort anchors by token position
        anchors = _get_sorted_token_anchors(
            tokens=tokens,
            anchor_positions=anchor_positions
        )

        ### 2. Calculate midpoint boundaries between anchors
        boundaries = _get_midpoint_boundaries(
            anchors=anchors,
            num_tokens=len(tokens)
        )

        ### 3. Expand each anchor up to B tokens, favoring balance
        regions = []

        for i, anchor in enumerate(anchors):

            region = _expand_single_anchor(
                anchor=anchor,
                left_boundary=boundaries[i],
                right_boundary=boundaries[i + 1],
                B=B
            )

            regions.append(region)

        ### Store expanded regions and original document information
        results.append({
            "document_id": document_id,
            "document_text": document_text,
            "tokens": tokens,
            "anchor_positions": anchor_positions,
            "regions": regions
        })

    return results

def _get_sorted_token_anchors(
    tokens: list[tuple[str, int, int]],
    anchor_positions: dict[str, list[int]]
) -> list[int]:

    ### Map character start positions to token indices
    char_to_token = {
        start: i
        for i, (_, start, end) in enumerate(tokens)
    }

    ### Convert anchor positions
    anchors = [
        char_to_token[position]
        for positions in anchor_positions.values()
        for position in positions
        if position in char_to_token
    ]

    return sorted(set(anchors))

def _get_midpoint_boundaries(
    anchors: list[int],
    num_tokens: int
) -> list[int]:

    if not anchors:
        return [0]

    boundaries = [0]

    for i in range(len(anchors) - 1):
        midpoint = (anchors[i] + anchors[i + 1] + 1) // 2
        boundaries.append(midpoint)

    boundaries.append(num_tokens)

    return boundaries

def _expand_single_anchor(
    anchor: int,
    left_boundary: int,
    right_boundary: int,
    B: int
) -> tuple[int, int]:

    ### Maximum length allowed within the boundaries
    region_length = min(B, right_boundary - left_boundary)

    ### Initially center the region around the anchor
    start = anchor - (region_length - 1) // 2

    ### Shift the region if it exceeds a boundary
    start = max(left_boundary, start)
    start = min(start, right_boundary - region_length)

    end = start + region_length

    return (start, end)

def _region_scores(region, tokens, query_terms, CS_Weight=0.7, DS_Weight=0.3):
    """Shared relevance: distinct-term coverage plus anchor-occurrence density."""
    if CS_Weight < 0 or DS_Weight < 0 or abs(CS_Weight + DS_Weight - 1) > 1e-9:
        raise ValueError("Nonnegative weights must sum to 1")
    start, end = region
    terms = set(query_terms)
    words = [token[0] for token in tokens[start:end]]
    coverage = len(set(words) & terms) / len(terms) if terms else 0.0
    density = sum(word in terms for word in words) / len(words) if words else 0.0
    return coverage, density, CS_Weight * coverage + DS_Weight * density

def merge_anchoring_regions(
    expanded_documents: list[dict[str, int | list[tuple[int, int]]]],
    documents_and_anchor_positions: list[dict],
    query_terms: list[str],
    B: int,
    CS_Weight: float = 0.7,
    DS_Weight: float = 0.3
) -> list[dict[str, int | list[tuple[int, int]]]]:
    """
    Merges neighboring expanded regions within each document.

    Steps:
        1. Sort regions by their starting token positions.
        2. Identify query terms contained in each region.
        3. Check whether neighboring regions can merge:
            a. Combined region length <= B.
            b. Next region introduces a new query term OR
               increases the relevance score.
        4. Merge qualifying regions.

    B = Maximum number of tokens per merged region.

    Output:
    [
        {
            "document_id": 0,
            "regions": [
                (5, 31),
                (35, 49)
            ]
        }
    ]
    """

    if B < 1:
        raise ValueError("B must be positive")

    results = []

    ### Map original documents by ID
    documents_by_id = {
        document["document_id"]: document
        for document in documents_and_anchor_positions
    }

    for expanded_document in expanded_documents:

        ### Extract document information
        document_id = expanded_document["document_id"]
        regions = sorted(
            expanded_document["regions"],
            key=lambda region: region[0]
        )

        document = documents_by_id[document_id]
        tokens = document["tokens"]

        ### 1. Identify query terms in each region
        region_query_terms = [
            _get_query_terms_in_region(
                region=region,
                tokens=tokens,
                query_terms=query_terms
            )
            for region in regions
        ]

        ### 2. Merge neighboring regions
        merged_regions = []

        if regions:

            current_region = regions[0]
            current_terms = region_query_terms[0].copy()

            for i in range(1, len(regions)):

                next_region = regions[i]
                next_terms = region_query_terms[i]

                ### Check merging conditions
                should_merge = _should_merge_regions(
                    current_region=current_region,
                    next_region=next_region,
                    current_query_terms=current_terms,
                    next_query_terms=next_terms,
                    B=B, tokens=tokens, query_terms=query_terms,
                    CS_Weight=CS_Weight, DS_Weight=DS_Weight
                )

                if should_merge:

                    ### Merge token intervals
                    current_region = (
                        current_region[0],
                        max(current_region[1], next_region[1])
                    )

                    ### Update query terms represented
                    current_terms.update(next_terms)

                else:

                    ### Store completed region
                    merged_regions.append(current_region)

                    ### Start new region
                    current_region = next_region
                    current_terms = next_terms.copy()

            merged_regions.append(current_region)

        results.append({
            "document_id": document_id,
            "document_text": document["document_text"],
            "tokens": document["tokens"],
            "anchor_positions": document["anchor_positions"],
            "regions": merged_regions
        })

    return results

def _get_query_terms_in_region(
    region: tuple[int, int],
    tokens: list[tuple[str, int, int]],
    query_terms: list[str]
) -> set[str]:

    start, end = region

    ### Extract words in region
    region_words = {
        token[0]
        for token in tokens[start:end]
    }

    ### Return query terms found in region
    return region_words.intersection(query_terms)

def _should_merge_regions(
    current_region, next_region, current_query_terms, next_query_terms,
    B, tokens, query_terms, CS_Weight=0.7, DS_Weight=0.3
) -> bool:
    """Merge touching/overlapping regions if coverage or relevance improves."""
    combined = (min(current_region[0], next_region[0]),
                max(current_region[1], next_region[1]))
    if next_region[0] > current_region[1] or combined[1] - combined[0] > B:
        return False
    if next_query_terms - current_query_terms:
        return True
    current_score = _region_scores(current_region, tokens, query_terms,
                                   CS_Weight, DS_Weight)[2]
    merged_score = _region_scores(combined, tokens, query_terms,
                                  CS_Weight, DS_Weight)[2]
    return merged_score > current_score + 1e-12

def rank_candidate_spans(
    merged_documents: list[dict],
    query_terms: list[str],
    CS_Weight: float = 0.7,
    DS_Weight: float = 0.3
) -> list[dict]:
    """
    Ranks candidate spans using query coverage and anchor density.

    Score(S) = CS_Weight * C(S) + DS_Weight * D(S)

    C(S) = Distinct query terms in S / Total distinct query terms
    D(S) = Matched query-term occurrences in S / Token length of S

    Default weights:
        CS_Weight = 0.7
        DS_Weight = 0.3

    Output:
    [
        {
            "document_id": 0,
            "regions": [
                {
                    "start": 5,
                    "end": 31,
                    "coverage_score": 0.75,
                    "density_score": 0.1154,
                    "score": 0.5596
                }
            ]
        }
    ]
    """

    if CS_Weight < 0 or DS_Weight < 0:
        raise ValueError("Weights cannot be negative")

    if not abs(CS_Weight + DS_Weight - 1.0) < 1e-9:
        raise ValueError("Weights must sum to 1")

    results = []
    distinct_query_terms = set(query_terms)

    for document in merged_documents:

        ### Extract document information
        document_id = document["document_id"]
        tokens = document["tokens"]
        regions = document["regions"]

        ranked_regions = []

        ### Calculate scores for each candidate span
        for start, end in regions:

            coverage_score, density_score, score = _region_scores(
                (start, end), tokens, query_terms, CS_Weight, DS_Weight
            )

            ranked_regions.append({
                "start": start,
                "end": end,
                "coverage_score": coverage_score,
                "density_score": density_score,
                "score": score
            })

        ### Rank regions from highest to lowest score
        ranked_regions.sort(
            key=lambda region: region["score"],
            reverse=True
        )

        ### Store results
        results.append({
            "document_id": document_id,
            "regions": ranked_regions
        })

    return results

def select_spans_within_context_budget(
    ranked_documents: list[dict],
    merged_documents: list[dict],
    query_terms: list[str],
    context_budget_tokens: int
) -> list[dict]:
    """
    Selects the highest-ranked candidate spans within a total
    context token budget.

    Steps:
        1. Combine ranked spans from all documents.
        2. Sort spans by relevance score (descending).
        3. Select fitting spans, skipping those that exceed the remaining budget.
        4. Extract text and matched query terms.

    Output:
    [
        {
            "document_id": 0,
            "text": "in 1915, einstein completed ...",
            "matched_terms": ["theory", "general", "relativity"]
        }
    ]
    """

    if context_budget_tokens < 1:
        raise ValueError("Context budget must be positive")

    ### Map documents by ID
    documents_by_id = {
        document["document_id"]: document
        for document in merged_documents
    }

    ### 1. Combine ranked spans from all documents
    all_spans = []

    for document in ranked_documents:

        document_id = document["document_id"]

        for region in document["regions"]:

            all_spans.append({
                "document_id": document_id,
                **region
            })

    ### 2. Sort spans by relevance score
    all_spans.sort(
        key=lambda span: span["score"],
        reverse=True
    )

    ### 3. Select spans within context budget
    selected_spans = []
    tokens_used = 0

    for span in all_spans:

        start = span["start"]
        end = span["end"]
        span_length = end - start

        ### Skip oversized spans so smaller candidates can still fit
        if tokens_used + span_length > context_budget_tokens:
            continue

        document_id = span["document_id"]
        document = documents_by_id[document_id]

        tokens = document["tokens"]
        document_text = document["document_text"]

        ### 4. Extract original text
        start_char = tokens[start][1]
        end_char = tokens[end - 1][2]

        span_text = document_text[start_char:end_char]

        ### Identify matched query terms
        matched_terms = _get_query_terms_in_region(
            region=(start, end),
            tokens=tokens,
            query_terms=query_terms
        )

        selected_spans.append({
            "document_id": document_id,
            "start_token": start,
            "end_token": end,
            "start_char": start_char,
            "end_char": end_char,
            "text": span_text,
            "matched_terms": sorted(matched_terms),
            "relevance_score": span["score"],
            "token_count": span_length
        })

        tokens_used += span_length

    return selected_spans

def baseline_documents(documents, B, adaptive=False):
    """Query-independent fixed or sentence-packed chunk construction."""
    result = []
    for doc_id, text in enumerate(clean_all_documents(documents)):
        tokens = tokenize_document(text)
        if adaptive:
            # Sentence boundaries are mapped to the same word-token positions.
            ends = [i + 1 for i, token in enumerate(tokens)
                    if i == len(tokens) - 1 or re.search(
                        r'[.!?]', text[token[2]:tokens[i + 1][1]])]
            units, start = [], 0
            for end in ends:
                units.extend((j, min(j + B, end)) for j in range(start, end, B))
                start = end
            regions = []
            for start, end in units:
                if regions and end - regions[-1][0] <= B:
                    regions[-1] = (regions[-1][0], end)
                else:
                    regions.append((start, end))
        else:
            regions = [(i, min(i + B, len(tokens))) for i in range(0, len(tokens), B)]
        result.append(dict(document_id=doc_id, document_text=text, tokens=tokens,
                           regions=regions, anchor_positions={}))
    return result

def answer_scores(prediction, references):
    """SQuAD-style normalized exact match and multiset token F1."""
    def normalize(text):
        text = ''.join(c for c in text.lower() if c not in string.punctuation)
        return re.sub(r'\b(a|an|the)\b', ' ', text).split()
    predicted = normalize(prediction)
    scores = []
    for reference in references:
        gold = normalize(reference)
        common = sum((Counter(predicted) & Counter(gold)).values())
        f1 = 2 * common / (len(predicted) + len(gold)) if predicted and gold else float(predicted == gold)
        scores.append((float(predicted == gold), f1))
    return {'exact_match': max(s[0] for s in scores),
            'answer_f1': max(s[1] for s in scores)}

def evidence_scores(spans, evidence):
    retrieved = {(s['document_id'], i) for s in spans
                 for i in range(s['start_token'], s['end_token'])}
    gold = {(s['document_id'], i) for s in evidence
            for i in range(s['start_token'], s['end_token'])}
    matched = len(retrieved & gold)
    return dict(evidence_recall=matched / len(gold) if gold else 0.0,
                precision=matched / len(retrieved) if retrieved else 0.0,
                completeness=int(bool(gold) and gold <= retrieved),
                context_tokens=sum(s['token_count'] for s in spans))

def ollama_answer(prompt, base_url, model):
    payload = dict(model=model, prompt=prompt, stream=False,
                   options=dict(temperature=0, seed=42, num_predict=64, num_ctx=4096))
    request = urllib.request.Request(base_url.rstrip('/') + '/api/generate',
                                     json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)['response'].strip()

def ollama_config(base_url=None, model=None, env_path=None):
    """Resolve explicit arguments > environment > project .env (KEY=value)."""
    path = Path(env_path) if env_path is not None else Path(__file__).with_name('.env')
    values = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, separator, value = line.partition('=')
            if not separator:
                raise ValueError('Expected KEY=value in .env')
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key.strip()] = value
    base_url = (base_url if base_url is not None else os.environ.get(
        'OLLAMA_BASE_URL', values.get('OLLAMA_BASE_URL', ''))).strip()
    model = (model if model is not None else os.environ.get(
        'OLLAMA_MODEL', values.get('OLLAMA_MODEL', ''))).strip()
    if not base_url or not model:
        raise ValueError('Set OLLAMA_BASE_URL and OLLAMA_MODEL in .env or the environment. '
                         'Copy example.env to .env and fill in both values.')
    from urllib.parse import urlsplit
    parsed = urlsplit(base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('OLLAMA_BASE_URL must be an HTTP or HTTPS URL')
    return base_url.rstrip('/'), model


def evaluate_example(base_url=None, model=None, output_dir='evaluation_results'):
    query, document_text = EXAMPLE_QUERY, EXAMPLE_DOCUMENT
    base_url, model = ollama_config(base_url, model)
    source = document_text.lower()
    evidence_texts = [
        'albert einstein was a german-born theoretical physicist.',
        'in 1915, einstein completed the theory of general relativity, which describes gravity as the curvature of spacetime.'
    ]
    tokens = tokenize_document(source)
    evidence = []
    for text in evidence_texts:
        start = source.index(text)
        indices = [i for i, (_, a, b) in enumerate(tokens) if a >= start and b <= start + len(text)]
        evidence.append(dict(document_id=0, start_token=min(indices), end_token=max(indices)+1, text=text))
    terms = extract_query_terms_from_query(query)
    no_match = not any(t[0] in terms for t in tokens)
    configurations = [(B, budget) for B in (128, 256, 512) for budget in (512, 1024)] + [(32, 64)]
    rows, answer_cache = [], {}
    for B, budget in configurations:
        for method in ('Query-Anchored', 'Fixed-Size', 'Sentence-Adaptive'):
            preprocessing_ms = 0.0
            if method != 'Query-Anchored':
                start = perf_counter()
                candidates = baseline_documents([source], B, method == 'Sentence-Adaptive')
                preprocessing_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            if method == 'Query-Anchored':
                spans = query_anchored_expansion(query, [source], B, budget)
            else:
                ranked = rank_candidate_spans(candidates, terms)
                spans = select_spans_within_context_budget(ranked, candidates, terms, budget)
            latency_ms = (perf_counter() - start) * 1000
            # Restore source order for a coherent prompt, identically for all methods.
            ordered = sorted(spans, key=lambda s: (s['document_id'], s['start_token']))
            context = '\n'.join(s['text'] for s in ordered)
            prompt = ('Answer the question using only the context. Return only the short answer, '
                      'without explanation. If the context does not answer it, return UNKNOWN.\n\n'
                      f'Context:\n{context}\n\nQuestion: {query}\nAnswer:')
            prediction, error, metrics = None, None, dict(exact_match=None, answer_f1=None)
            if model:
                try:
                    if prompt not in answer_cache:
                        answer_cache[prompt] = ollama_answer(prompt, base_url, model)
                    prediction = answer_cache[prompt]
                    metrics = answer_scores(prediction, ['Albert Einstein', 'Einstein'])
                except Exception as exc:
                    error = str(exc)
            rows.append(dict(method=method, chunk_limit=B, context_budget=budget,
                             **evidence_scores(spans, evidence), **metrics,
                             no_query_match=no_match, preprocessing_ms=preprocessing_ms,
                             retrieval_ms=latency_ms, prediction=prediction,
                             generation_status='complete' if prediction is not None else ('error' if error else 'pending'),
                             generation_error=error, selected_spans=spans, prompt=prompt))
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    report = dict(question=query, references=['Albert Einstein', 'Einstein'],
                  evidence=evidence, model=model, base_url=base_url,
                  generation_options=dict(temperature=0, seed=42, num_predict=64, num_ctx=4096),
                  note='Single example; sentence-adaptive is a simplified structural baseline. Identical prompts reuse one model response.',
                  results=rows)
    (output / 'example_results.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    header = '| Method | B | Budget | Recall | Precision | Complete | EM | F1 | Tokens | Prep ms | Retrieval ms |'
    lines = ['# Example evaluation', '', query, '', report['note'], '',
             'Word-token metrics; N/A means no successful model response. Single-run timings.', '',
             header, '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [row[k] for k in ('method', 'chunk_limit', 'context_budget', 'evidence_recall',
                  'precision', 'completeness', 'exact_match', 'answer_f1', 'context_tokens',
                  'preprocessing_ms', 'retrieval_ms')]
        lines.append('| ' + ' | '.join('N/A' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v) for v in values) + ' |')
    (output / 'example_results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))
    return report

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query-anchored expansion example")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run and save the example comparison, including Ollama answers")
    args = parser.parse_args()
    if args.evaluate:
        try:
            evaluate_example()
        except ValueError as exc:
            parser.error(str(exc))
    else:
        spans = query_anchored_expansion(
            EXAMPLE_QUERY, [EXAMPLE_DOCUMENT], max_chunk_tokens=32,
            context_budget_tokens=64
        )
        print("Question:", EXAMPLE_QUERY)
        print("Query terms:", ", ".join(extract_query_terms_from_query(EXAMPLE_QUERY)))
        print("\nSelected context spans:")
        print(json.dumps(spans, indent=2))


if __name__ == "__main__":
    main()
