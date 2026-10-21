"""Regression tests for the main Python implementation."""
import unittest
import csv
import contextlib
import tempfile
from pathlib import Path
import adaptive_baseline
from tests.test_adaptive_baseline import FixtureBackend
import io
import json
from unittest.mock import patch
import main

ns = vars(main)


class DeliverableTests(unittest.TestCase):
    def test_example_exports_csv_for_success_and_failure(self):
        for error in (None, OSError('offline, try again\nnext line')):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                chunker = adaptive_baseline.AdaptiveChunker(FixtureBackend())
                with patch.object(adaptive_baseline, 'get_adaptive_chunker', return_value=chunker), \
                     patch.object(main, 'ollama_answer', return_value='Albert Einstein', side_effect=error), \
                     contextlib.redirect_stdout(io.StringIO()):
                    report = main.evaluate_example('http://example.test', 'mock-model', directory)
                output = Path(directory)
                self.assertFalse(list(output.glob('*.md')))
                with (output / 'example_results.csv').open(newline='', encoding='utf-8-sig') as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 21)
                for actual, expected in zip(rows, report['results']):
                    self.assertEqual(actual['question'], report['question'])
                    self.assertEqual(actual['prompt'], expected['prompt'])
                    self.assertEqual(json.loads(actual['selected_spans']), expected['selected_spans'])
                    self.assertEqual(float(actual['retrieval_ms']), expected['retrieval_ms'])
                    self.assertEqual(actual['generation_status'], 'error' if error else 'complete')
                    self.assertEqual(actual['exact_match'], '' if error else '1.0')
                    self.assertEqual(actual['generation_error'], str(error) if error else '')

    def test_ollama_disables_thinking(self):
        response = io.BytesIO(json.dumps(dict(response=' Albert Einstein ', done=True,
                                             done_reason='stop')).encode())
        with patch.object(main.urllib.request, 'urlopen', return_value=response) as post:
            answer = main.ollama_answer('Question', 'http://example.test/', 'model')
        self.assertEqual(answer, 'Albert Einstein')
        request = post.call_args.args[0]
        self.assertEqual(request.full_url, 'http://example.test/api/generate')
        payload = json.loads(request.data)
        self.assertIs(payload['think'], False)
        self.assertNotIn('think', payload['options'])

    def test_ollama_rejects_incomplete_or_empty_answers(self):
        cases = [
            (dict(response='Thinking Process: If the', done=True,
                  done_reason='length', eval_count=64), "done_reason='length', eval_count=64"),
            (dict(response='Einstein', done=False), 'incomplete'),
            (dict(response='  ', done=True, done_reason='stop'), 'empty answer'),
        ]
        for result, message in cases:
            with self.subTest(result=result):
                response = io.BytesIO(json.dumps(result).encode())
                with patch.object(main.urllib.request, 'urlopen', return_value=response):
                    with self.assertRaisesRegex(ValueError, message):
                        main.ollama_answer('Question', 'http://example.test', 'model')

    def test_configuration_precedence(self):
        with patch.dict(main.os.environ, {'OLLAMA_MODEL': 'environment-model'}, clear=True), \
             patch.object(main.Path, 'is_file', return_value=True), \
             patch.object(main.Path, 'read_text', return_value='# Settings\nOLLAMA_BASE_URL="http://example.test:11434/"\nOLLAMA_MODEL=file-model\n'):
            self.assertEqual(main.ollama_config(), ('http://example.test:11434', 'environment-model'))
            self.assertEqual(main.ollama_config(model='explicit-model')[1], 'explicit-model')

    def test_configuration_required(self):
        with patch.dict(main.os.environ, {}, clear=True), \
             patch.object(main.Path, 'is_file', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Copy example.env'):
                main.ollama_config()
            with self.assertRaisesRegex(ValueError, 'HTTP or HTTPS'):
                main.ollama_config('not-a-url', 'model')

    def test_terms_depend_on_question(self):
        extract = ns['extract_query_terms_from_query']
        self.assertEqual(extract('Where was Marie Curie born?'), ['marie', 'curie', 'born'])
        self.assertEqual(extract('THE theory, theory?'), ['theory'])
        self.assertEqual(extract('Which is the?'), [])

    def test_no_anchors(self):
        self.assertEqual(ns['query_anchored_expansion']('Which is the?', ['the theory']), [])
        self.assertEqual(ns['query_anchored_expansion']('volcano', ['the theory']), [])

    def test_merge_relevance_and_constraints(self):
        tokens = ns['tokenize_document']('alpha filler filler alpha alpha alpha')
        merge = ns['_should_merge_regions']
        args = dict(current_region=(0, 3), next_region=(3, 6),
                    current_query_terms={'alpha'}, next_query_terms={'alpha'},
                    B=6, tokens=tokens, query_terms=['alpha'])
        self.assertTrue(merge(**args))  # Same term, higher occurrence density.
        self.assertFalse(merge(**{**args, 'B': 5}))
        self.assertFalse(merge(**{**args, 'next_region': (4, 6)}))
        self.assertFalse(merge(**{**args, 'CS_Weight': 1, 'DS_Weight': 0}))
        self.assertTrue(merge(**{**args, 'next_query_terms': {'beta'}}))

    def test_answer_metrics(self):
        scores = ns['answer_scores']
        self.assertEqual(scores('The Albert Einstein.', ['Albert Einstein'])['exact_match'], 1)
        self.assertAlmostEqual(scores('Einstein', ['Albert Einstein'])['answer_f1'], 2/3)
        self.assertEqual(scores('Newton', ['Albert Einstein'])['answer_f1'], 0)

    def test_evidence_is_position_based(self):
        spans = [dict(document_id=0, start_token=0, end_token=2, token_count=2)]
        labels = [dict(document_id=0, start_token=1, end_token=3)]
        result = ns['evidence_scores'](spans, labels)
        self.assertEqual((result['evidence_recall'], result['precision'], result['completeness']), (.5, .5, 0))
        self.assertEqual(ns['evidence_scores']([], labels)['evidence_recall'], 0)

    def test_budgets_and_offsets(self):
        text = 'alpha beta filler ' * 40
        for B in (1, 7, 32):
            for budget in (1, 9, 64):
                spans = ns['query_anchored_expansion']('alpha beta', [text], B, budget)
                self.assertLessEqual(sum(s['token_count'] for s in spans), budget)
                for span in spans:
                    self.assertLessEqual(span['token_count'], B)
                    self.assertEqual(span['text'], text[span['start_char']:span['end_char']])
        for adaptive in (False,):  # Adaptive bounds are tested with its own backend fixture.
            docs = ns['baseline_documents'](['One short. ' + 'word '*40 + '. End.'], 7, adaptive)
            regions = docs[0]['regions']
            self.assertTrue(all(b-a <= 7 for a,b in regions))
            self.assertEqual([i for a,b in regions for i in range(a,b)], list(range(len(docs[0]['tokens']))))

    def test_skip_span_that_does_not_fit(self):
        docs = ns['baseline_documents'](['one two three four five six'], 6)
        ranked = [dict(document_id=0, regions=[dict(start=0,end=5,score=1),dict(start=5,end=6,score=.5)])]
        spans = ns['select_spans_within_context_budget'](ranked,docs,[],1)
        self.assertEqual(spans[0]['text'],'six')


if __name__ == '__main__':
    unittest.main()
