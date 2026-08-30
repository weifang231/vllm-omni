# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.queue_control import RequestSchedulingMetadata


def test_scheduling_metadata_survives_engine_request_roundtrip() -> None:
    request = OmniEngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        scheduling_metadata=RequestSchedulingMetadata(
            request_class="interactive",
            path="audio",
            deadline_monotonic_s=12.5,
        ),
    )

    encoded = MsgpackEncoder().encode(request)
    decoded = MsgpackDecoder(OmniEngineCoreRequest).decode(encoded)

    assert decoded.scheduling_metadata == RequestSchedulingMetadata(
        request_class="interactive",
        path="audio",
        deadline_monotonic_s=12.5,
    )
