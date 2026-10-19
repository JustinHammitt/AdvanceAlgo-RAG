"""Regression tests for the main Python implementation."""
import unittest
from unittest.mock import patch
import main

ns = vars(main)


class DeliverableTests(unittest.TestCase):
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
