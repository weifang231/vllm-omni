"""Real GPU sampler plus the actual post-sampler extension control path."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vllm_omni/benchmarks"))
from reference_replay_worker import ReferenceReplayWorkerExtension


class Runner:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.vllm_config = SimpleNamespace(speculative_config=None)
        self.model_config = SimpleNamespace(get_vocab_size=lambda: 16, is_hybrid=False)
        self.max_num_reqs = 4
        self.positions = torch.tensor([1, 1, 1, 1], device=self.device)
        self.query_start_loc = SimpleNamespace(gpu=torch.arange(5, device=self.device, dtype=torch.int32))
        self.input_batch = SimpleNamespace(req_ids=[], num_prompt_tokens=np.array([2] * 4))
        self.optimistic_seq_lens_cpu = torch.zeros(4, dtype=torch.int32)
        self.requests = {}
        self.calls = 0
        self.sampler = Sampler()

    def _sample(self, logits, spec):
        self.calls += 1
        n = logits.shape[0]
        meta = SamplingMetadata(
            temperature=None,
            all_greedy=True,
            all_random=False,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(n, device=self.device),
            presence_penalties=torch.zeros(n, device=self.device),
            repetition_penalties=torch.ones(n, device=self.device),
            output_token_ids=[[] for _ in range(n)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors([]),
        )
        return self.sampler(logits, meta)

    def add(self, rid, key, tokens):
        params = SamplingParams(max_tokens=len(tokens), extra_args={"reference_replay_key": key} if key else None)
        self.requests[rid] = SimpleNamespace(prompt_token_ids=[1, 2], sampling_params=params)

    def sample(self, ids, positions):
        self.input_batch.req_ids = ids
        self.positions[: len(ids)] = torch.tensor(positions, device=self.device)
        self.optimistic_seq_lens_cpu[: len(ids)] = torch.tensor(positions) + 1
        return (
            self._sample(torch.arange(16, device=self.device, dtype=torch.float32).repeat(len(ids), 1), None)
            .sampled_token_ids[:, 0]
            .tolist()
        )


class Worker(ReferenceReplayWorkerExtension):
    def __init__(self):
        self.model_runner = Runner()


def refs():
    return {
        key: {"prompt_token_ids": [1, 2], "token_ids": ids}
        for key, ids in {"a": [0, 5, 15], "b": [8, 15], "c": [11, 15]}.items()
    }


@unittest.skipUnless(torch.cuda.is_available(), "requires the assigned CUDA GPU")
class PostSamplerTests(unittest.TestCase):
    def test_actual_sampler_reorder_add_remove_and_exact_restoration(self):
        w = Worker()
        r = w.model_runner
        original = r._sample.__func__
        w.reference_replay_configure("replay", refs())
        for key, data in refs().items():
            r.add("internal-" + key, key, data["token_ids"])
        r.add("untracked", None, [1, 2])
        self.assertEqual(r.sample(["internal-a", "internal-b"], [1, 1]), [0, 8])
        self.assertEqual(r.sample(["internal-b", "internal-a"], [2, 2]), [15, 5])
        self.assertIsNot(r._reference_replay_controller.row_slots, r._reference_replay_controller.canonical_row_slots)
        self.assertEqual(r.sample(["internal-c", "internal-a", "untracked"], [1, 3, 1]), [11, 15, 15])
        audit = w.reference_replay_finish()
        self.assertEqual(r.calls, 3)
        self.assertEqual(audit["out_of_range"], 0)
        self.assertEqual(audit["valid_reference_lookups"], 6)
        self.assertEqual(audit["common_offset_view_calls"], 2)
        self.assertEqual(audit["fused_calls"], 1)
        w.reference_replay_configure("native")
        self.assertNotIn("_sample", r.__dict__)
        self.assertIs(r._sample.__func__, original)
        self.assertEqual(r.sample(["internal-a"], [1]), [15])

    def test_partial_prefill_tail_and_exhaustion_have_explicit_counters(self):
        w = Worker()
        r = w.model_runner
        w.reference_replay_configure("replay", {"a": refs()["a"]})
        r.add("internal-a", "a", [0, 5, 15])
        self.assertEqual(r.sample(["internal-a"], [0]), [15])
        self.assertEqual(r.sample(["internal-a"], [1]), [0])
        self.assertEqual(r.sample(["internal-a"], [4]), [15])
        a = w.reference_replay_finish()
        self.assertEqual(a["partial_prefill_samples_left_unchanged"], 1)
        self.assertEqual(a["max_tokens_tail_samples_left_unchanged"], 1)
        r.sample(["internal-a"], [5])
        with self.assertRaisesRegex(RuntimeError, "boundary"):
            w.reference_replay_finish()
        with self.assertRaisesRegex(RuntimeError, "previous replay audit"):
            w.reference_replay_configure("native")

    def test_noop_keeps_original_tokens_and_reads_reference(self):
        w = Worker()
        r = w.model_runner
        w.reference_replay_configure("noop", {"a": refs()["a"]})
        r.add("internal-a", "a", [0, 5, 15])
        self.assertEqual(r.sample(["internal-a"], [2]), [15])
        self.assertIs(r._reference_replay_controller.row_slots, r._reference_replay_controller.canonical_row_slots)
        self.assertEqual(w.reference_replay_finish()["reference_token_checksum"], 5)

    def test_wrong_identity_and_logprobs_fail(self):
        for failure in ["key", "prompt", "logprobs"]:
            w = Worker()
            r = w.model_runner
            w.reference_replay_configure("replay", {"a": refs()["a"]})
            r.add("internal-a", "a", [0, 5, 15])
            if failure == "key":
                r.requests["internal-a"].sampling_params.extra_args["reference_replay_key"] = "foreign"
            elif failure == "prompt":
                r.requests["internal-a"].prompt_token_ids = [2, 1]
            else:
                r.requests["internal-a"].sampling_params.logprobs = 0
            with self.assertRaises(ValueError):
                r.sample(["internal-a"], [1])
            with self.assertRaisesRegex(RuntimeError, "previously failed"):
                w.reference_replay_finish()

    def test_mixed_offsets_and_original_exception(self):
        w = Worker()
        r = w.model_runner
        w.reference_replay_configure("replay", refs())
        r.add("internal-a", "a", [0, 5, 15])
        r.add("internal-b", "b", [8, 15])
        self.assertEqual(r.sample(["internal-a", "internal-b"], [2, 1]), [5, 8])
        self.assertEqual(w.reference_replay_finish()["fused_calls"], 1)

        def fail(*args):
            raise RuntimeError("original sampler failure")

        r._reference_replay_original = fail
        with self.assertRaisesRegex(RuntimeError, "original sampler failure"):
            r.sample(["internal-a"], [1])
        with self.assertRaisesRegex(RuntimeError, "previously failed"):
            w.reference_replay_finish()

    def test_cached_numpy_tracks_inplace_updates_and_tensor_replacement(self):
        w = Worker()
        r = w.model_runner
        w.reference_replay_configure("replay", {"a": refs()["a"]})
        r.add("internal-a", "a", [0, 5, 15])
        c = r._reference_replay_controller
        old_tensor, old_numpy = c.seq_lens_tensor, c.seq_lens_numpy
        self.assertEqual(r.sample(["internal-a"], [1]), [0])
        self.assertEqual(r.sample(["internal-a"], [2]), [5])
        self.assertIs(c.seq_lens_numpy, old_numpy)
        self.assertEqual(old_numpy[0], 3)
        r.optimistic_seq_lens_cpu = torch.zeros(4, dtype=torch.int32)
        self.assertEqual(r.sample(["internal-a"], [3]), [15])
        self.assertIsNot(c.seq_lens_tensor, old_tensor)
        self.assertIs(c.seq_lens_tensor, r.optimistic_seq_lens_cpu)
        self.assertIsNot(c.seq_lens_numpy, old_numpy)
        self.assertEqual(w.reference_replay_finish()["valid_reference_lookups"], 3)
        self.assertEqual(r.calls, 3)

    def test_cached_frames_canonical_then_reordered_have_distinct_ownership(self):
        w = Worker()
        r = w.model_runner
        references = {"a": refs()["a"], "b": {"prompt_token_ids": [1, 2], "token_ids": [8, 9, 15]}}
        w.reference_replay_configure("replay", references)
        for key, row in references.items():
            r.add("internal-" + key, key, row["token_ids"])
        c = r._reference_replay_controller
        original_table = c.table.clone()
        self.assertEqual(r.sample(["internal-a", "internal-b"], [1, 1]), [0, 8])
        self.assertIs(c.batch_frames, c.canonical_frames)
        first_frame = c.batch_frames[0]
        self.assertEqual(r.sample(["internal-b", "internal-a"], [2, 2]), [9, 5])
        self.assertIsNot(c.batch_frames, c.canonical_frames)
        self.assertEqual(first_frame[:, 0].tolist(), [0, 8])
        self.assertTrue(torch.equal(c.table, original_table))
        self.assertEqual(w.reference_replay_finish()["valid_reference_lookups"], 4)
        self.assertEqual(r.calls, 2)

    def test_prompt_array_replacement_refreshes_memoryview(self):
        w = Worker()
        r = w.model_runner
        w.reference_replay_configure("replay", {"a": refs()["a"]})
        r.add("internal-a", "a", [0, 5, 15])
        c = r._reference_replay_controller
        self.assertEqual(r.sample(["internal-a"], [1]), [0])
        old = c.prompt_array
        r.input_batch.num_prompt_tokens = r.input_batch.num_prompt_tokens.copy()
        self.assertEqual(r.sample(["internal-a"], [2]), [5])
        self.assertIsNot(c.prompt_array, old)
        self.assertIs(c.prompt_array, r.input_batch.num_prompt_tokens)
        self.assertEqual(w.reference_replay_finish()["valid_reference_lookups"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
