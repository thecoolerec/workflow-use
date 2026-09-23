import unittest

from browser_use.llm.messages import UserMessage
from pydantic import BaseModel
from workflow_use.llm.factory import create_chat_model
from workflow_use.llm.grok_build import ChatGrokBuild


class ExampleOutput(BaseModel):
	value: str


class StubGrokBuild(ChatGrokBuild):
	async def _invoke_cli(self, prompt, schema):
		return {
			'text': '',
			'structured_output': {'value': 'ok'},
			'stopReason': 'end_turn',
			'usage': {
				'input_tokens': 10,
				'cache_read_input_tokens': 5,
				'cache_creation_input_tokens': 0,
				'output_tokens': 3,
				'total_tokens': 18,
			},
		}


class GrokBuildAdapterTests(unittest.IsolatedAsyncioTestCase):
	async def test_structured_output_is_validated(self):
		llm = StubGrokBuild()
		result = await llm.ainvoke([UserMessage(content='Return a value')], output_format=ExampleOutput)

		self.assertEqual(result.completion, ExampleOutput(value='ok'))
		self.assertEqual(result.stop_reason, 'end_turn')
		self.assertIsNotNone(result.usage)
		self.assertEqual(result.usage.total_tokens, 18)

	async def test_plain_text_output(self):
		class PlainStub(ChatGrokBuild):
			async def _invoke_cli(self, prompt, schema):
				return {'text': 'hello', 'stopReason': 'end_turn'}

		result = await PlainStub().ainvoke([UserMessage(content='Say hello')])
		self.assertEqual(result.completion, 'hello')

	def test_factory_creates_grok_build_provider(self):
		llm = create_chat_model('grok-build')
		self.assertIsInstance(llm, ChatGrokBuild)
		self.assertEqual(llm.provider, 'grok-build')


if __name__ == '__main__':
	unittest.main()
