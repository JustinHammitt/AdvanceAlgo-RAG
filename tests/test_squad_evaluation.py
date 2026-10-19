"""Offline SQuAD tests; no downloads or real model evaluation."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main
import squad_evaluation as squad
import adaptive_baseline
from tests.test_adaptive_baseline import FixtureBackend


def dataset():
    articles = []
    for index in range(10):
        articles.append(dict(title=f'Article {index}', paragraphs=[dict(
            context='Einstein developed relativity. Other unrelated words.',
            qas=[dict(id=f'{index}-{i:02}', question='Who developed relativity?',
                      answers=[dict(text='Einstein', answer_start=0)]) for i in range(30)]),
            dict(context='Distractor paragraph.', qas=[])]))
    return articles


class SquadTests(unittest.TestCase):
    def setUp(self):
        self.chunker = adaptive_baseline.AdaptiveChunker(FixtureBackend())
        patcher = patch.object(adaptive_baseline, 'get_adaptive_chunker', return_value=self.chunker)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_reproducible_article_disjoint_selection(self):
        questions = squad.select_questions(dataset())
        self.assertEqual(questions, squad.select_questions(list(reversed(dataset()))))
        self.assertEqual(len({q['id'] for q in questions}), 300)
        dev = [q for q in questions if q['split'] == 'dev']
        test = [q for q in questions if q['split'] == 'test']
        self.assertEqual((len(dev), len(test)), (60, 240))
        self.assertFalse({q['title'] for q in dev} & {q['title'] for q in test})
        self.assertNotEqual([q['id'] for q in questions], [q['id'] for q in squad.select_questions(dataset(), seed=7)])

    def test_rejects_invalid_data(self):
        with self.assertRaisesRegex(ValueError, 'ten articles'):
            squad.select_questions(dataset()[:9])
        articles = dataset()
        articles[0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] = 5
        with self.assertRaisesRegex(ValueError, 'offset'):
            squad.select_questions(articles)

    def test_evidence_and_unicode_offsets(self):
        question = dict(id='unicode', document_id=0, documents=['\u0130. Einstein discovered it. Other.'],
                        answers=[dict(answer_start=3, text='Einstein')])
        labels = squad.sentence_evidence(question)
        self.assertEqual(labels, [dict(document_id=0, start_char=3, end_char=27)])
        tokens = squad.token_evidence(question, labels)
        self.assertEqual(tokens, [dict(document_id=0, start_token=1, end_token=4)])
        with self.assertRaisesRegex(ValueError, 'range'):
            squad.token_evidence(question, [dict(document_id=0, start_char=0, end_char=999)])

    def test_offline_runner_writes_real_scores_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'squad.json'
            path.write_text(json.dumps(dict(version='1.1', data=dataset())), encoding='utf-8')
            with patch.object(main, 'ollama_config', return_value=('http://example.test', 'mock-model')), \
                 patch.object(main, 'ollama_answer', return_value='Einstein') as generate, \
                 contextlib.redirect_stdout(io.StringIO()):
                output = squad.evaluate_squad(path, split='test', limit=2, output_root=directory)
            rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 36)
            self.assertTrue(all(r['exact_match'] == 1 for r in rows))
            self.assertTrue(all(r['split'] == 'test' for r in rows))
            self.assertTrue(all(sum(s['token_count'] for s in r['selected_spans']) <= r['context_budget'] for r in rows))
            self.assertLess(generate.call_count, 36)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(len(manifest['selection']), 300)
            self.assertEqual(len(manifest['run_question_ids']), 2)
            self.assertEqual(manifest['evidence_kind'], 'answer_sentence_proxy')
            self.assertIn('Adaptive-Chunking', manifest['methods'])
            audit = json.loads((output / 'adaptive_selection.json').read_text())
            self.assertTrue(audit)
            self.assertEqual(set(audit[0]['candidates'][0]['metrics']), set(adaptive_baseline.METRICS))
            self.assertTrue((output / 'summary.md').exists())
            with (output / 'summary.csv').open(newline='', encoding='utf-8-sig') as file:
                exported = list(csv.DictReader(file))
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(len(exported), len(summary))
            for actual, expected in zip(exported, summary):
                self.assertEqual(actual['method'], expected['method'])
                self.assertEqual(float(actual['answer_f1']), expected['answer_f1'])
                self.assertEqual(float(actual['evidence_recall']), expected['evidence_recall'])
                self.assertEqual(int(actual['generation_errors']), expected['generation_errors'])
            self.assertEqual(len(json.loads((output / 'preprocessing.json').read_text())), 6)

    def test_generation_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'squad.json'
            path.write_text(json.dumps(dict(version='1.1', data=dataset())), encoding='utf-8')
            with patch.object(main, 'ollama_config', return_value=('http://example.test', 'mock')), \
                 patch.object(main, 'ollama_answer', side_effect=OSError('offline')), \
                 contextlib.redirect_stdout(io.StringIO()):
                output = squad.evaluate_squad(path, limit=1, output_root=directory)
            rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['answer_f1'] is None and r['generation_status'] == 'error' for r in rows))
            self.assertTrue(all(r['answer_f1'] == 0 and r['generation_errors'] == 1 for r in squad.summarize(rows)))

    def test_cli_routes_preserve_existing_modes(self):
        with patch('sys.argv', ['main.py']), patch.object(main, 'evaluate_example') as example, \
             patch.object(squad, 'evaluate_squad') as evaluate, contextlib.redirect_stdout(io.StringIO()):
            main.main()
            example.assert_not_called()
            evaluate.assert_not_called()
        with patch('sys.argv', ['main.py', '--evaluate']), patch.object(main, 'evaluate_example') as example:
            main.main()
            example.assert_called_once_with()
        with patch('sys.argv', ['main.py', '--squad-evaluate', '--squad-limit', '2']), \
             patch.object(squad, 'evaluate_squad') as evaluate:
            main.main()
            self.assertEqual(evaluate.call_args.kwargs['limit'], 2)


if __name__ == '__main__':
    unittest.main()
