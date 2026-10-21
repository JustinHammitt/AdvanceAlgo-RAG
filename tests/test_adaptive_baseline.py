"""Offline tests for selection, offsets, caching, and the upstream adapter."""
import unittest
from unittest.mock import Mock, patch

import adaptive_baseline as adaptive
import main


class FixtureBackend:
    """Synthetic test-only scores; never used by production evaluation."""
    def __init__(self):
        self.analyses = 0
        self.scores = 0

    def describe(self):
        return {'backend': 'test-fixture'}

    def analyze(self, text):
        self.analyses += 1
        return dict(sentence_starts=[0] + [m.end() for m in adaptive.re.finditer(r'\.\s+', text)],
                    block_starts=[m.end() for m in adaptive.re.finditer(r'\n\n', text)], pairs=[])

    def split(self, text, target, minimum):
        tokens = adaptive.tokens_of(text)
        regions = [(i, min(i+target, len(tokens))) for i in range(0, len(tokens), target)]
        return adaptive.chunks_from_regions(text, tokens, regions)

    def score(self, text, chunks, analysis, minimum, B):
        self.scores += 1
        # A controllable preference for short chunks proves selection is not
        # hardcoded to the first candidate.
        value = 1.0 if max(len(adaptive.tokens_of(c)) for c in chunks) <= max(1, B//2) else .2
        return dict.fromkeys(adaptive.METRICS, value)


class AdaptiveTests(unittest.TestCase):
    def test_device_setting_from_env_file_and_shell_override(self):
        with patch.dict(adaptive.os.environ, {}, clear=True), \
             patch.object(main.Path, 'is_file', return_value=True), \
             patch.object(main.Path, 'read_text', return_value='ADAPTIVE_DEVICE="cuda:1"\n'):
            self.assertEqual(adaptive.adaptive_settings()['ADAPTIVE_DEVICE'], 'cuda:1')
            with patch.dict(adaptive.os.environ, {'ADAPTIVE_DEVICE': 'cpu'}):
                self.assertEqual(adaptive.adaptive_settings()['ADAPTIVE_DEVICE'], 'cpu')
        with patch.dict(adaptive.os.environ, {}, clear=True), \
             patch.object(main.Path, 'is_file', return_value=False):
            self.assertEqual(adaptive.adaptive_settings()['ADAPTIVE_DEVICE'], 'cpu')

    def test_device_validation(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 2
        self.assertEqual(adaptive.validate_device('cpu', torch), 'cpu')
        torch.cuda.is_available.assert_not_called()
        self.assertEqual(adaptive.validate_device('cuda', torch), 'cuda:0')
        self.assertEqual(adaptive.validate_device('cuda:1', torch), 'cuda:1')
        for device in ('gpu', 'cuda:-1', 'cuda:2', ''):
            with self.subTest(device=device), self.assertRaises(ValueError):
                adaptive.validate_device(device, torch)
        torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            adaptive.validate_device('cuda:0', torch)

    def test_selects_best_strategy_and_records_five_scores(self):
        chunker = adaptive.AdaptiveChunker(FixtureBackend())
        document = chunker.build_documents(['one two three four five six seven eight'], 4)[0]
        audit = document['adaptive_selection']
        self.assertEqual(audit['strategy'], 'recursive_half')
        self.assertEqual(audit['mean_score'], 1)
        self.assertEqual(len(audit['candidates']), 4)
        self.assertTrue(all(set(c['metrics']) == set(adaptive.METRICS) for c in audit['candidates']))

    def test_offsets_coverage_and_hard_caps_for_every_candidate(self):
        chunker = adaptive.AdaptiveChunker(FixtureBackend())
        for text in ['one two. three four five.\n\nsix seven eight nine.', 'İ longword repeated repeated. '*9]:
            for B in (1, 3, 8, 128):
                doc = chunker.build_documents([text], B)[0]
                for candidate in doc['adaptive_selection']['candidates']:
                    regions = candidate['regions']
                    self.assertTrue(all(0 < b-a <= B for a,b in regions))
                    self.assertEqual([i for a,b in regions for i in range(a,b)], list(range(len(doc['tokens']))))
                    self.assertEqual(''.join(adaptive.chunks_from_regions(doc['document_text'], doc['tokens'], regions)), text.lower())

    def test_empty_documents_and_invalid_budget(self):
        chunker = adaptive.AdaptiveChunker(FixtureBackend())
        self.assertEqual(chunker.build_documents(['', '...'], 8)[0]['regions'], [])
        self.assertEqual(chunker.backend.analyses, 0)
        with self.assertRaisesRegex(ValueError, 'positive'):
            chunker.build_documents(['a'], 0)

    def test_analysis_reused_across_budgets_and_ids_not_cached(self):
        backend = FixtureBackend()
        chunker = adaptive.AdaptiveChunker(backend)
        chunker.build_documents(['one two three four'], 4)
        count = backend.scores
        docs = chunker.build_documents(['one two three four']*2, 4)
        self.assertEqual([d['document_id'] for d in docs], [0,1])
        self.assertEqual(backend.scores, count)
        chunker.build_documents(['one two three four'], 2)
        self.assertEqual(backend.analyses, 1)

    def test_null_metrics_are_explicit_and_ties_are_stable(self):
        backend = FixtureBackend()
        backend.score = lambda *args: dict(RC=None, ICC=None, DCC=None, BI=1., SC=1.)
        result = adaptive.AdaptiveChunker(backend).build_documents(['one two'], 4)[0]
        audit = result['adaptive_selection']
        self.assertEqual(audit['strategy'], 'recursive_full')
        self.assertEqual(audit['candidates'][0]['applicable_metrics'], ['BI','SC'])

    def test_missing_scores_and_nonfinite_scores_fail(self):
        for values in [dict(RC=1), dict.fromkeys(adaptive.METRICS, float('nan'))]:
            backend = FixtureBackend()
            backend.score = lambda *args: values
            with self.assertRaises(ValueError):
                adaptive.AdaptiveChunker(backend).build_documents(['one two'], 4)

    def test_adapter_invokes_real_metric_interfaces(self):
        backend = adaptive.PaperBackend.__new__(adaptive.PaperBackend)
        backend.metrics, backend.embedder = Mock(), Mock()
        m = backend.metrics
        m.compute_filtered_missing_ref_error.return_value = .25
        m.compute_intrachunk_cohesion.return_value = .5
        m.compute_contextual_coherence.return_value = .6
        m.compute_block_integrity.return_value = .7
        m.compute_size_compliance.return_value = 1.
        scores = backend.score('one two', ['one ', 'two'],
                               dict(pairs=[[(0,2),(4,6)]], sentence_starts=[0,4], block_starts=[]), 1, 4)
        self.assertEqual(scores, dict(RC=.75, ICC=.5, DCC=.6, BI=.7, SC=1.))
        self.assertEqual(m.compute_contextual_coherence.call_args.kwargs['window_context_tokens'], 3000)
        self.assertEqual(m.compute_size_compliance.call_args.kwargs['max_tokens'], 4)

    def test_public_baseline_routes_to_selector(self):
        chunker = adaptive.AdaptiveChunker(FixtureBackend())
        with patch.object(adaptive, 'get_adaptive_chunker', return_value=chunker):
            docs = main.baseline_documents(['one two three four'], 4, adaptive=True)
        self.assertIn('adaptive_selection', docs[0])


if __name__ == '__main__':
    unittest.main()
