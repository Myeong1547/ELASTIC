"""Maps vLLM request IDs (strings) to small integer seq_ids — the
sequence axis of the NVMe blob layout (`BlockKVLayout`: block j of
sequence seq has global number k = seq·n + j (0 ≤ j < n, immutable blocks)
or k = N_seq·n + (j − n)·N_seq + seq (j ≥ n, mutable blocks) by Eqs. 10–11,
with n = ⌊S/block_size⌋, `n_immutable` (S = prompt length; the code takes
`prompt_len // block_size`), and a block's buffer position is the layout's
closed form of k). vLLM's physical block ids are never a position here.

Ids are assigned once per batch by the gate at the batch's prefill step, in
row order (vLLM's flat packing order): 0..N_seq-1. seq_id is the outer
(seq-major) axis of the immutable range on disk, so row order is file order.
"""

from __future__ import annotations

from typing import Iterable


class SeqIdAllocator:
    def __init__(self):
        # req_id (vLLM string) → seq_id (int)
        self._req_to_seq: dict[str, int] = {}

    def assign(self, req_ids: Iterable[str]) -> None:
        """Hand the batch ids 0..N_seq-1 in caller order, never by iterating a
        `set`. seq_id orders the immutable range of every block region on
        disk (seq 0's blocks, then seq 1's), and the zig-zag prefill's GPU
        batches are contiguous slices of the step's rows, so row order makes a
        span's blocks one range of global numbers k — which is what the
        save's closed-form runs rest on. The caller is the gate, which passes
        the rows' request ids in row order. A `set` iterates in string-hash
        order, randomised per process, which would scatter every span over
        the block region."""
        self._req_to_seq = {r: i for i, r in enumerate(req_ids)}

    def get(self, req_id: str) -> int:
        return self._req_to_seq[req_id]
