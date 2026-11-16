"""`from_units` is the one place a cell is priced: the endpoint context
S+gen, T_fwd as one token's dense ops plus S+gen keys of attention, kv_gen
per token, and the record's acceptance rule."""
import pytest

from elastic.cost.markers import (ARRIVAL_DRAIN_TOKENS_DEFAULT,
                                  KVGEN_STORE_SHAPE, LAYER_STEP_SHAPE,
                                  LOAD_CHUNK_TOKENS_DEFAULT)
from elastic.policy.from_units import check_units_record, units_to_solver_inputs

D, L, S, GEN = 9216, 64, 1024, 128
BS = 16
_RUN = dict(slots_mem_bytes=96.0 * 2**30, beta=0.5, block_size=BS)


def _record(**over):
    u = {"read_GBps": 12.8, "write_GBps": 1.82,
         "T_kv_gen_per_token_s": 4.7e-6, "T_attn_per_token_s": 2.6e-7,
         "T_dense_per_token_s": 3.0e-6,
         "S_op": S, "gen_len": GEN, "weight_bytes": 2.0e9,
         "provenance": {"layer_step_shape": LAYER_STEP_SHAPE,
                        "kvgen_store_shape": KVGEN_STORE_SHAPE,
                        "kvgen_wave_tokens": LOAD_CHUNK_TOKENS_DEFAULT,
                        "drain_tokens": ARRIVAL_DRAIN_TOKENS_DEFAULT}}
    prov = over.pop("provenance", None)
    u.update(over)
    if prov:
        u["provenance"].update(prov)
    return u


@pytest.fixture(autouse=True)
def _run_env(monkeypatch):
    monkeypatch.setenv("ELASTIC_LOAD_CHUNK_TOKENS", str(LOAD_CHUNK_TOKENS_DEFAULT))
    monkeypatch.setenv("ELASTIC_ARRIVAL_DRAIN_TOKENS",
                       str(ARRIVAL_DRAIN_TOKENS_DEFAULT))


@pytest.mark.parametrize("n_seq", [64, 128, 256, 512])
def test_prices_at_the_endpoint_per_token(n_seq):
    u = _record()
    inp = units_to_solver_inputs(u, n_seq=n_seq, S=S, d_model=D, num_layers=L,
                                 gen_len=GEN, read_GBps=13.4, write_GBps=1.82,
                                 **_RUN)
    end = S + GEN
    assert inp.T_fwd == pytest.approx(
        u["T_dense_per_token_s"] + end * u["T_attn_per_token_s"])
    assert inp.T_kv_gen == pytest.approx(u["T_kv_gen_per_token_s"] * end)
    assert inp.t_kv == pytest.approx(end * 4 * D / 13.4e9)
    assert inp.t_save_kv == pytest.approx(BS * 4 * D / 1.82e9)
    assert inp.t_w == pytest.approx(u["weight_bytes"] / 13.4e9)
    assert inp.BW_NVMe == pytest.approx(12.8e9)       # the measured rate


@pytest.mark.parametrize("bad", [
    dict(gen_len=64),                                       # another context
    dict(S_op=1025),
    dict(provenance={"layer_step_shape": None}),            # explicit None
    dict(provenance={"layer_step_shape": "other-v0"}),      # another shape
    dict(provenance={"kvgen_wave_tokens": 8192}),           # another launch size
    dict(provenance={"drain_tokens": 16384}),               # another call volume
    dict(provenance={"kvgen_store_shape": None}),           # explicit None
    dict(provenance={"kvgen_store_shape": "other-v0"}),     # another shape
    dict(weight_bytes=0),
])
def test_refuses_a_record_that_prices_something_else(bad):
    with pytest.raises(ValueError):
        check_units_record(_record(**bad), S=S, gen_len=GEN)
    with pytest.raises(ValueError):
        units_to_solver_inputs(_record(**bad), n_seq=128, S=S, d_model=D,
                               num_layers=L, gen_len=GEN, **_RUN)


def test_accepts_its_own_cell():
    check_units_record(_record(), S=S, gen_len=GEN)
