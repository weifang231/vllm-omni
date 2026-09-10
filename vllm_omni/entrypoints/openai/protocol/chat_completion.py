from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse, ChatCompletionStreamResponse
from vllm_omni.outputs.termination import TerminationEvidence


class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    modality: str | None = "text"
    metrics: dict[str, Any] | None = None
    termination: TerminationEvidence | None = None


class OmniChatCompletionResponse(ChatCompletionResponse):
    metrics: dict[str, Any] | None = None
