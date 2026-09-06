"""Keep structured options on AstrBot OpenAI requests without global mutation."""
from __future__ import annotations

from copy import copy, deepcopy
from typing import Any


async def generate_analysis_request(context: Any, kwargs: dict[str, Any]) -> Any:
    try:
        provider = context.get_provider_by_id(kwargs["chat_provider_id"])
    except (AttributeError, KeyError, TypeError):
        provider = None
    if (provider is None
            or type(provider).__module__ != "astrbot.core.provider.sources.openai_source"
            or type(provider).__name__ != "ProviderOpenAIOfficial"
            or not isinstance(getattr(provider, "provider_config", None), dict)
            or not callable(getattr(getattr(provider, "client", None), "with_options", None))):
        return await context.llm_generate(**kwargs)

    # AstrBot 4.26.7's _prepare_chat_payload accepts but discards **kwargs.
    # Its supported custom_extra_body path is applied at the SDK boundary.
    # Clone the provider and SDK facade; share only the HTTP connection pool.
    # Native key rotation must not change the registered client's api_key.
    scoped = copy(provider)
    scoped.provider_config = deepcopy(provider.provider_config)
    scoped.client = provider.client.with_options()
    extras = scoped.provider_config.get("custom_extra_body")
    extras = dict(extras) if isinstance(extras, dict) else {}
    for key in ("response_format", "temperature", "enable_thinking"):
        if key in kwargs:
            extras[key] = deepcopy(kwargs[key])
    scoped.provider_config["custom_extra_body"] = extras
    payload = {key: value for key, value in kwargs.items() if key != "chat_provider_id"}
    return await scoped.text_chat(**payload)
