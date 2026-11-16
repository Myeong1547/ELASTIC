"""The ACT read geometry: an ACT block region's ranks [0, n) cut into read
segments at the segment size and, apart from them, into kv_gen launches at
the launch size (ELASTIC_LOAD_CHUNK_TOKENS).

`act_segments_and_launches` decides both, and each is a silent failure mode
— kv_gen is told only the launch's BASE buffer block (`mem_act_base + w0`,
w0 the launch's first rank), so a launch that was not a dense slice of the
block region would write real K/V into the wrong cache blocks and the tokens
would just be wrong; a launch issued before a segment it overlaps landed
would recompute K/V from stale activation-buffer bytes.
"""
import random

import pytest

from elastic.vllm_runtime.loader_thread import act_segments_and_launches

KB = 4096          # bytes per block (block_bytes), arbitrary here
BASE = 1 << 20     # the block region's file base


def _check(n_blocks, segment_blocks, launch_blocks):
    segments, launches = act_segments_and_launches(n_blocks, BASE, KB, segment_blocks, launch_blocks)
    want = list(range(n_blocks))
    # 1. the segments read every block exactly once, at its closed-form
    #    offset, in ascending rank order, and none is over the segment size;
    #    the block region is cut every segment_blocks from rank 0
    got = []
    for (off, n, r0) in segments:
        assert off == BASE + r0 * KB
        assert 1 <= n <= segment_blocks
        got += [r0 + i for i in range(n)]
    assert got == want
    assert [d[2] for d in segments] == list(range(0, n_blocks, segment_blocks))
    # 2. the kv_gen launches cover every block exactly once, ascending; each
    #    is one dense slice, cut every launch_blocks from rank 0
    got = []
    for (w0, n, _ids) in launches:
        assert 1 <= n <= launch_blocks
        got += [w0 + i for i in range(n)]
    assert got == want
    assert [w[0] for w in launches] == list(range(0, n_blocks, launch_blocks))
    # 3. a launch names exactly the segments that overlap it
    for (w0, n, ids) in launches:
        over = [i for i, (_o, dn, d0) in enumerate(segments)
                if d0 < w0 + n and w0 < d0 + dn]
        assert ids == over
    return segments, launches


def test_one_run_one_segment_one_launch_when_it_fits():
    segments, launches = _check(8, segment_blocks=16, launch_blocks=16)
    assert len(segments) == 1 and len(launches) == 1 and launches[0][2] == [0]


def test_the_kv_gen_launch_is_cut_apart_from_the_segment():
    # 175b-like: a 16 MiB segment holds 42 blocks, a 1024-token launch 64
    # (the default launch is 2048 tokens; the cut does not depend on it).
    # The launches stay 64 blocks; the first spans the first two segments.
    segments, launches = _check(200, segment_blocks=42, launch_blocks=64)
    assert [d[1] for d in segments] == [42, 42, 42, 42, 32]
    assert [w[1] for w in launches] == [64, 64, 64, 8]
    assert launches[0][2] == [0, 1]
    assert launches[1][2] == [1, 2, 3]


def test_a_segment_holds_several_launches():
    # 6.7b-like at a 1024-token launch: a 16 MiB segment is 128 blocks, two
    # launches of 64.
    segments, launches = _check(300, segment_blocks=128, launch_blocks=64)
    assert [d[1] for d in segments] == [128, 128, 44]
    assert [w[2] for w in launches] == [[0], [0], [1], [1], [2]]


@pytest.mark.parametrize("seed", range(40))
def test_random_shapes_keep_every_property(seed):
    rng = random.Random(seed)
    _check(rng.randint(1, 400), segment_blocks=rng.randint(1, 50),
           launch_blocks=rng.randint(1, 50))
