import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from advisor.provider_request import generate_analysis_request


class ProviderOpenAIOfficial:
    __module__ = "astrbot.core.provider.sources.openai_source"

    def __init__(self):
        self.provider_config = {"custom_extra_body": {"existing": 1}, "key": ["test-key"]}
        self.client = SimpleNamespace(api_key="registered-key")
        self.client.with_options = lambda: SimpleNamespace(api_key=self.client.api_key)

    async def text_chat(self, **kwargs):
        self.client.api_key = "request-key"
        await asyncio.sleep(0)
        return deepcopy(self.provider_config["custom_extra_body"])


class ProviderRequestTests(IsolatedAsyncioTestCase):
    async def test_concurrent_native_requests_preserve_options_without_shared_mutation(self):
        provider = ProviderOpenAIOfficial()
        before = deepcopy(provider.provider_config)
        context = SimpleNamespace(get_provider_by_id=lambda _: provider, llm_generate=AsyncMock())
        first, second = await asyncio.gather(*[
            generate_analysis_request(context, {"chat_provider_id": "native", "prompt": "synthetic",
                "response_format": {"type": "json_schema", "name": label}, "enable_thinking": False})
            for label in ("first", "second")
        ])
        self.assertEqual(first["response_format"]["name"], "first")
        self.assertEqual(second["response_format"]["name"], "second")
        self.assertIs(first["enable_thinking"], False)
        self.assertEqual(provider.provider_config, before)
        self.assertEqual(provider.client.api_key, "registered-key")
        context.llm_generate.assert_not_called()

    async def test_other_providers_keep_the_framework_entry_point(self):
        context = SimpleNamespace(get_provider_by_id=lambda _: object(), llm_generate=AsyncMock(return_value="ok"))
        kwargs = {"chat_provider_id": "other", "prompt": "synthetic", "temperature": 0}
        self.assertEqual(await generate_analysis_request(context, kwargs), "ok")
        context.llm_generate.assert_awaited_once_with(**kwargs)
