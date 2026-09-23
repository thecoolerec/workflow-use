from __future__ import annotations

import asyncio

import aiofiles
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, overload

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import AssistantMessage, BaseMessage
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatGrokBuild(BaseChatModel):
	"""Browser Use chat-model adapter backed by the locally authenticated Grok Build CLI.

	Each invocation is intentionally stateless. Grok Build is used only as the model
	backend; Browser Use remains responsible for browser actions.
	"""

	model: str = 'default'
	executable: str = 'grok'
	cwd: str | Path | None = None
	max_turns: int = 1
	reasoning_effort: str | None = None
	timeout_seconds: float = 180.0

	_verified_api_keys: bool = True
	supports_vision: bool = False

	@property
	def provider(self) -> str:
		return 'grok-build'

	@property
	def name(self) -> str:
		return self.model if self.model != 'default' else 'grok-build-default'

	def _message_text(self, message: BaseMessage) -> str:
		content = message.content
		if isinstance(content, str):
			text = content
		else:
			parts: list[str] = []
			for part in content or []:
				part_type = getattr(part, 'type', None)
				if part_type == 'text':
					parts.append(getattr(part, 'text', ''))
				elif part_type == 'refusal':
					parts.append(f"[Refusal] {getattr(part, 'refusal', '')}")
				elif part_type == 'image_url':
					parts.append('[Image omitted: ChatGrokBuild currently runs Browser Use with vision disabled]')
			text = '\n'.join(part for part in parts if part)

		if isinstance(message, AssistantMessage) and message.tool_calls:
			tool_calls = [tool_call.model_dump(mode='json') for tool_call in message.tool_calls]
			text = f'{text}\nTool calls: {json.dumps(tool_calls, ensure_ascii=False)}'.strip()

		return text

	def _build_prompt(self, messages: list[BaseMessage]) -> str:
		conversation: list[str] = [
			'You are acting as a stateless chat-completion backend for Browser Use.',
			'Do not use Grok Build tools, shell commands, filesystem access, web search, MCP, or subagents.',
			'Do not inspect the current working directory. Respond only from the conversation below.',
			'Follow SYSTEM messages as the highest-priority instructions in the supplied conversation.',
			'',
		]
		for message in messages:
			conversation.append(f'<{message.role.upper()}>')
			conversation.append(self._message_text(message))
			conversation.append(f'</{message.role.upper()}>')
		return '\n'.join(conversation)

	def _usage_from_result(self, data: dict[str, Any]) -> ChatInvokeUsage | None:
		usage = data.get('usage')
		if not isinstance(usage, dict):
			return None

		input_tokens = int(usage.get('input_tokens') or 0)
		cached_tokens = int(usage.get('cache_read_input_tokens') or 0)
		cache_creation_tokens = int(usage.get('cache_creation_input_tokens') or 0)
		completion_tokens = int(usage.get('output_tokens') or usage.get('completion_tokens') or 0)
		total_tokens = int(
			usage.get('total_tokens')
			or input_tokens + cached_tokens + cache_creation_tokens + completion_tokens
		)

		return ChatInvokeUsage(
			prompt_tokens=input_tokens + cached_tokens + cache_creation_tokens,
			prompt_cached_tokens=cached_tokens or None,
			prompt_cache_creation_tokens=cache_creation_tokens or None,
			prompt_image_tokens=None,
			completion_tokens=completion_tokens,
			total_tokens=total_tokens,
		)

	def _parse_json_text(self, text: str) -> Any:
		candidate = text.strip()
		if candidate.startswith('```'):
			lines = candidate.splitlines()
			if len(lines) >= 3:
				candidate = '\n'.join(lines[1:-1]).strip()
		return json.loads(candidate)

	async def _invoke_cli(self, prompt: str, schema: dict[str, Any] | None) -> dict[str, Any]:
		if shutil.which(self.executable) is None:
			raise ModelProviderError(
				message=(
					f'Grok Build executable "{self.executable}" was not found in PATH. '
					'Install/login to Grok Build first, then verify "grok -p \"Reply OK\"" works.'
				),
				model=self.name,
			)

		run_cwd = Path(self.cwd) if self.cwd is not None else Path(tempfile.gettempdir()) / 'workflow-use-grok-build'
		run_cwd.mkdir(parents=True, exist_ok=True)

		prompt_path: Path | None = None
		try:
			with tempfile.NamedTemporaryFile(
				mode='w',
				encoding='utf-8',
				suffix='.txt',
				prefix='workflow-use-grok-',
				delete=False,
			) as prompt_file:
				prompt_file.write(prompt)
				prompt_path = Path(prompt_file.name)

			cmd = [
				self.executable,
				'--prompt-file',
				str(prompt_path),
				'--output-format',
				'json',
				'--verbatim',
				'--rules',
				(
					'Act only as a stateless chat-completion backend. Do not call tools. '
					'Treat <SYSTEM> blocks in the supplied prompt as system-level instructions.'
				),
				'--no-plan',
				'--no-subagents',
				'--no-ask-user',
				'--disable-web-search',
				'--max-turns',
				str(self.max_turns),
				'--no-auto-update',
			]
			if self.model and self.model != 'default':
				cmd.extend(['--model', self.model])
			if self.reasoning_effort:
				cmd.extend(['--reasoning-effort', self.reasoning_effort])
			if schema is not None:
				schema_json = json.dumps(schema, ensure_ascii=False, separators=(',', ':'))
				# Windows has a finite process command-line limit. Large Browser Use
				# action schemas are moved into the prompt file instead of argv.
				if os.name == 'nt' and len(schema_json) > 24000:
					async with aiofiles.open(prompt_path, 'a', encoding='utf-8') as prompt_file:
						await prompt_file.write(
							'\\n\\nReturn ONLY a JSON object that validates against this JSON Schema:\\n'
							+ schema_json
						)
					schema = None
				else:
					cmd.extend(['--json-schema', schema_json])

			env = os.environ.copy()
			env['GROK_DISABLE_AUTOUPDATER'] = '1'
			env['GROK_MEMORY'] = '0'

			process = await asyncio.create_subprocess_exec(
				*cmd,
				cwd=str(run_cwd),
				env=env,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
			)
			try:
				stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout_seconds)
			except TimeoutError as exc:
				process.kill()
				await process.communicate()
				raise ModelProviderError(
					message=f'Grok Build timed out after {self.timeout_seconds:g}s.',
					model=self.name,
				) from exc

			stdout_text = stdout.decode('utf-8', errors='replace').strip()
			stderr_text = stderr.decode('utf-8', errors='replace').strip()

			if process.returncode != 0:
				hint = ''
				if 'json-schema' in stderr_text.lower():
					hint = ' Grok Build structured output requires a recent build with --json-schema support.'
				raise ModelProviderError(
					message=f'Grok Build exited with code {process.returncode}: {stderr_text or stdout_text}.{hint}',
					model=self.name,
				)

			try:
				data = json.loads(stdout_text)
			except json.JSONDecodeError as exc:
				raise ModelProviderError(
					message=f'Grok Build returned invalid JSON: {stdout_text[:1000]}',
					model=self.name,
				) from exc

			if not isinstance(data, dict):
				raise ModelProviderError(message='Grok Build returned a non-object JSON response.', model=self.name)
			return data
		finally:
			if prompt_path is not None:
				try:
					prompt_path.unlink(missing_ok=True)
				except OSError:
					pass

	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any
	) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type[T] | None = None,
		**kwargs: Any,
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		prompt = self._build_prompt(messages)
		schema = (
			SchemaOptimizer.create_optimized_json_schema(output_format)
			if output_format is not None
			else None
		)
		data = await self._invoke_cli(prompt, schema)
		usage = self._usage_from_result(data)
		stop_reason = data.get('stopReason') or data.get('stop_reason')

		if output_format is None:
			completion = str(data.get('text') or '')
			return ChatInvokeCompletion(completion=completion, usage=usage, stop_reason=stop_reason)

		if 'structured_output' in data:
			structured = data['structured_output']
		else:
			structured = data.get('structuredOutput')
		if structured is None:
			text = str(data.get('text') or '')
			try:
				structured = self._parse_json_text(text)
			except (json.JSONDecodeError, TypeError) as exc:
				raise ModelProviderError(
					message='Grok Build did not return schema-validated structured_output.',
					model=self.name,
				) from exc

		try:
			completion = output_format.model_validate(structured)
		except Exception as exc:
			raise ModelProviderError(
				message=f'Grok Build structured output did not match {output_format.__name__}: {exc}',
				model=self.name,
			) from exc

		return ChatInvokeCompletion(completion=completion, usage=usage, stop_reason=stop_reason)
