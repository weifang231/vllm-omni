# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import thinker2talker_token_only

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source(request_id="ming-request", **completion):
    return SimpleNamespace(
        request_id=request_id,
        finished=True,
        outputs=[SimpleNamespace(**completion)],
    )


@pytest.mark.parametrize("last_delta", ["", ".", "。"])
def test_final_delta_uses_complete_thinker_answer(last_delta):
    answer = "The sky looks blue because air scatters sunlight."
    source = _source(text=last_delta, cumulative_text=answer)

    inputs = thinker2talker_token_only([source])

    assert len(inputs) == 1
    assert inputs[0]["additional_information"]["text"] == answer


def test_final_only_output_without_cumulative_text():
    answer = "春天到了，花朵盛开。"

    inputs = thinker2talker_token_only([_source(text=answer)])

    assert len(inputs) == 1
    assert inputs[0]["additional_information"]["text"] == answer


def test_unavailable_cumulative_text_uses_complete_text():
    answer = "Tea is ready."

    inputs = thinker2talker_token_only([_source(text=answer, cumulative_text=None)])

    assert inputs[0]["additional_information"]["text"] == answer


@pytest.mark.parametrize(
    "completion",
    [
        {},
        {"text": ""},
        {"text": " \n\t"},
        {"text": None},
        {"text": ".", "cumulative_text": ""},
        {"text": "last delta", "cumulative_text": " \n\t"},
    ],
)
def test_empty_completed_text_skips_talker_with_warning(completion, caplog):
    inputs = thinker2talker_token_only([_source(**completion)])

    assert inputs == []
    assert "ming-request" in caplog.text
    assert "skipping speech generation" in caplog.text


def test_empty_source_preserves_other_source_text_and_voice(caplog):
    sources = [
        _source("empty", text="", cumulative_text=""),
        _source("valid", text=".", cumulative_text="A complete answer."),
    ]
    prompts = [
        SimpleNamespace(additional_information={"voice_name": "unused"}),
        SimpleNamespace(additional_information={"voice_name": "DB30"}),
    ]

    inputs = thinker2talker_token_only(sources, prompts)

    assert len(inputs) == 1
    assert inputs[0]["additional_information"]["text"] == "A complete answer."
    assert inputs[0]["additional_information"]["voice_name"] == "DB30"
    assert "request empty" in caplog.text
