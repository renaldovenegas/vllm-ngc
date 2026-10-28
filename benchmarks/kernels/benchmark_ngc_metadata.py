# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure metadata preparation, including CPU kernel-submission overhead."""

import json
import statistics
import time

import torch

from vllm.v1.worker.block_table import MultiGroupBlockTable


def main():
    results = []
    for requests in (1, 8):
        for query_len in (1, 16, 256):
            groups = 14
            tokens = requests * query_len
            max_tokens = max(tokens, 256)
            tables = MultiGroupBlockTable(
                max_num_reqs=requests + 2,
                max_num_batched_tokens=max_tokens,
                pin_memory=True,
                device=torch.device("cuda"),
                block_sizes=[16] * groups,
                max_num_blocks=[160] * groups,
                batched_group_ids=tuple(range(groups)),
            )
            for row in range(requests):
                tables.add_row(tuple(list(range(1, 161)) for _ in range(groups)), row)
            tables.commit_block_table(requests)
            starts = torch.arange(requests + 1, device="cuda", dtype=torch.int32)
            starts *= query_len
            positions = torch.arange(2048, 2048 + query_len, device="cuda").repeat(
                requests
            )
            lengths = torch.full(
                (requests,), 2048 + query_len, device="cuda", dtype=torch.int32
            )
            evicted = (
                (torch.arange(groups, device="cuda", dtype=torch.int32) % 4 + 1) * 16
            ).repeat(requests, 1)
            indices = torch.arange(requests, device="cuda").repeat_interleave(query_len)

            def separate(
                tables=tables,
                requests=requests,
                starts=starts,
                positions=positions,
                evicted=evicted,
                indices=indices,
                lengths=lengths,
            ):
                retained = []
                for group, table in enumerate(tables.block_tables):
                    table.compute_slot_mapping(
                        requests, starts, positions - evicted[indices, group]
                    )
                    retained.append(lengths - evicted[:, group])
                return retained

            def fused(
                tables=tables,
                requests=requests,
                starts=starts,
                positions=positions,
                lengths=lengths,
                evicted=evicted,
            ):
                tables.compute_ngc_metadata(
                    requests, starts, positions, lengths, evicted
                )

            expected_lengths = separate()
            expected_slots = [
                table.slot_mapping.gpu.clone() for table in tables.block_tables
            ]
            fused()
            for group, table in enumerate(tables.block_tables):
                torch.testing.assert_close(
                    table.slot_mapping.gpu, expected_slots[group]
                )
                torch.testing.assert_close(
                    tables.ngc_retained.gpu[group, :requests], expected_lengths[group]
                )
            for _ in range(10):
                separate()
                fused()
            samples = {"separate": [], "fused": []}
            for trial in range(20):
                order = [("separate", separate), ("fused", fused)]
                if trial % 2:
                    order.reverse()
                for name, fn in order:
                    torch.accelerator.synchronize()
                    start = time.perf_counter()
                    for _ in range(10):
                        fn()
                    torch.accelerator.synchronize()
                    samples[name].append((time.perf_counter() - start) * 1e6 / 10)
            results.append(
                dict(
                    requests=requests,
                    query_len=query_len,
                    groups=groups,
                    **{
                        name + "_us": statistics.median(values)
                        for name, values in samples.items()
                    },
                )
            )
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "timing": "wall clock including CPU submission; warmed kernels",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
