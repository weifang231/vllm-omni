"""HTTP rendering must retain media until the engine binds a receiver replica."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.helpers.serving_chat import build_serving_chat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage_type", ["llm", "diffusion"])
async def test_chat_preserves_raw_media_for_engine_owned_cache(stage_type):
    serving = build_serving_chat()
    serving.engine_client.engine.get_stage_metadata.return_value = SimpleNamespace(stage_type=stage_type)
    serving._needs_multistage_multimodal_split = lambda: False
    serving._get_supported_speakers = lambda: set()
    renderer = MagicMock()
    media = object()
    raw = {"prompt_token_ids": [1, 2], "multi_modal_data": {"image": [media]}}
    rendered = {"type": "multimodal", "mm_kwargs": {"image": [None]}}
    renderer.render_messages_async = AsyncMock(return_value=([], raw))
    renderer.tokenize_prompts_async = AsyncMock(return_value=[raw])
    renderer.render_chat_async = AsyncMock(return_value=([[]], [rendered]))
    request = SimpleNamespace(
        build_tok_params=lambda _: "tokens",
        build_chat_params=lambda *_: SimpleNamespace(with_defaults=lambda *_, **__: "chat"),
        mm_processor_kwargs={"min_pixels": 256}, cache_salt="salt", modalities=["text"],
    )
    _, (prompt,) = await serving._preprocess_chat(request, [], None, "auto", renderer=renderer)
    assert prompt["mm_processor_kwargs"] == {"min_pixels": 256}
    assert prompt["cache_salt"] == "salt"
    if stage_type == "llm":
        assert prompt["multi_modal_data"]["image"][0] is media
        renderer.render_chat_async.assert_not_called()
    else:
        assert prompt["type"] == "multimodal"
        renderer.render_messages_async.assert_not_called()
