"""Gate check: token bit-identity, error beacons, and the write stream.

An arm's tokens are compared with the reference's first gen tokens per
sequence, gen being the arm's own configured gen_len (its CSV row), so every
sequence must carry exactly that many (a gen-1 arm is checked against a
longer reference's first token, and a sequence cut short fails). The two
zig-zag arms must also show the zig-zag prefill firing over the whole batch
in one step: its beacon's T equals the run's prompt tokens.

Token equality is necessary and not sufficient, so one arm runs with
ELASTIC_WRITE_AUDIT=1 and the checker reports the stream's shape: one
request per block region per event, covering the file-contiguous blocks the
layout's offsets give it (`sqe`, `desc` = events), and the largest request
(`max`), next to the NVMe engine's own item and SQE counts. The writeback
(WriteRouter) hands requests to the NVMe engine as built and the NVMe engine
composes them into unit-sized SQEs, so there is no fragmentation or oversize
rule — the numbers are printed so a shape change is visible.
"""
import csv, json, os, re
from itertools import zip_longest

RES = os.environ["GATE_RES"]
ARMS = {"SBIG": "B128", "SB05": "B64", "S10": "512", "S05": "512",
        "ST5": "515", "SW2": "512", "SW3": "512", "SD5": "512",
        "SBL5": "BL5", "SW5": "512", "SW7": "512",
        "SR2": "512", "SRL": "512", "SRL1": "512", "SRL5": "515",
        "SG1": "512"}
# the arm that carries ELASTIC_WRITE_AUDIT (see gate.sh)
AUDIT_ARM = "S05"
# the arms whose batch spans more than one prefill GPU batch (gate.sh)
ZIGZAG_ARMS = ("SBIG", "SB05")


def audit_shape(path):
    """-> ((n_sqe, worst, n_desc, eng_items, eng_sqes), None) over the
    submit records and the NVMe engine record."""
    try:
        recs = json.load(open(path))
    except Exception as e:
        return None, str(e)
    # ["S", kind, layer, offset, bytes, iovec segments, user data, event id,
    #  path]
    subs = [r for r in recs if r and r[0] == "S" and len(r) >= 8]
    if not subs:
        return None, "no submit records"
    worst = max(r[4] for r in subs)
    eng = [r for r in recs if r and r[0] == "E"]
    eng_items, eng_sqes = (int(eng[-1][1]), int(eng[-1][2])) if eng else (-1, -1)
    return (len(subs), worst, len({r[7] for r in subs}), eng_items,
            eng_sqes), None


fails = 0
for arm, r in ARMS.items():
    try:
        a = json.load(open(f"{RES}/ela_{arm}.json"))
        ref = json.load(open(f"{RES}/ref_{r}.json"))
        with open(f"{RES}/ela_{arm}.csv") as fh:
            rows = list(csv.DictReader(fh))
        gen = int(rows[0]["gen_len"])
    except Exception as e:
        print(f"{arm:5s} *** MISSING *** {e}"); fails += 1; continue
    ref = [x[:gen] for x in ref]
    ok = a == ref
    n = 0 if ok else sum(1 for x, y in zip_longest(a, ref) if x != y)
    try:
        bcn = open(f"{RES}/bcn_{arm}.txt", errors="ignore").read()
    except Exception:
        bcn = ""
    # A loader or writeback error ends that thread with its traceback in the
    # log.
    try:
        log = open(f"{RES}/ela_{arm}.log", errors="ignore").read()
    except Exception:
        log = ""
    n_bad = log.count("Traceback")
    # Beacon counts that show the decode ACT path ran: the layer-end save
    # of the activation blocks among the step's last N_seq blocks, from their
    # closed-form homes.
    n_tailw = len(re.findall(r"tail_write path=", bcn))
    n_tail = bcn.count("taild ")
    bad = []
    if n_bad:
        bad.append(f"TRACEBACKS x{n_bad}")
    # Every arrival-decode arm writes, at least once, a current block that is
    # an activation block; a zero here means the decode ACT path did not run
    # at all. SG1 (gen 1, no decode step) is not checked here.
    if arm != "SG1" and not n_tail:
        bad.append("decode ACT save NEVER FIRED")
    # A zig-zag arm's prefill must be the zig-zag prefill over the whole
    # batch in one step: the beacon's T is that step's token count.
    if arm in ZIGZAG_ARMS:
        m = re.search(r"layer-outer prefill ACTIVE T=(\d+)", log)
        ptoks = {int(row["prompt_toks"]) for row in rows}
        if m is None:
            bad.append("layer-outer prefill NEVER FIRED")
        elif ptoks != {int(m.group(1))}:
            bad.append(f"layer-outer prefill T={m.group(1)} "
                       f"!= prompt_toks {sorted(ptoks)}")
    extra = ""
    if arm == AUDIT_ARM:
        got, err = audit_shape(f"{RES}/waudit_{arm}.json")
        if err:
            bad.append(f"NO WRITE AUDIT ({err})")
        else:
            n_sqe, worst, n_desc, eng_items, eng_sqes = got
            # sqe/desc/max = the requests the writeback handed the NVMe
            # engine; eng = what the NVMe engine composed them into
            # (unit-sized SQEs).
            extra = (f" sqe={n_sqe} desc={n_desc} "
                     f"max={worst / 2**20:.2f}MiB "
                     f"eng_items={eng_items} eng_sqes={eng_sqes}")
    fails += (not ok) or bool(bad)
    print(f"{arm:5s} {'MATCH ' if ok else f'DIFFER {n:3d} '} "
          f"bad={n_bad} tail={n_tail} tailw={n_tailw}{extra}"
          + ("   " + " ".join(bad) if bad else ""))
print("GATE", "PASS" if fails == 0 else f"FAIL ({fails})")
