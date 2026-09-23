from browser_use.llm import ChatBrowserUse
from browser_use.llm.base import BaseChatModel

from workflow_use.llm.grok_build import ChatGrokBuild


def create_chat_model(provider: str, model: str | None = None) -> BaseChatModel:
	"""Create a Browser Use-compatible chat model for workflow generation."""
	provider_normalized = provider.strip().lower().replace('_', '-')

	if provider_normalized in {'grok-build', 'grok'}:
		return ChatGrokBuild(model=model or 'default')

	if provider_normalized in {'browser-use', 'browseruse', 'bu'}:
		return ChatBrowserUse(model=model or 'bu-latest')

	raise ValueError(
		f'Unsupported LLM provider: {provider}. Supported providers: browser-use, grok-build.'
	)
