import unittest

from workflow_use.healing.deterministic_converter import DeterministicWorkflowConverter


class DeterministicTerminalExtractTests(unittest.TestCase):
	def setUp(self):
		self.converter = DeterministicWorkflowConverter()

	def test_adds_terminal_extract_when_done_action_was_not_replayable(self):
		steps = [{'type': 'navigation', 'url': 'https://example.com'}]

		result = self.converter.ensure_terminal_extract(steps, 'Open example.com and read the page title')

		self.assertEqual(result[-1]['type'], 'extract_page_content')
		self.assertEqual(result[-1]['goal'], 'Open example.com and read the page title')

	def test_does_not_duplicate_existing_extract(self):
		steps = [
			{'type': 'navigation', 'url': 'https://example.com'},
			{'type': 'extract_page_content', 'goal': 'Read the page title'},
		]

		result = self.converter.ensure_terminal_extract(steps, 'Read the page title')

		self.assertEqual(len(result), 2)
		self.assertEqual(result[-1]['goal'], 'Read the page title')


if __name__ == '__main__':
	unittest.main()
