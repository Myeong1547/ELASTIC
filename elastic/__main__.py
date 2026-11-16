"""Entry points of the elastic pipeline, in the order they run:

    python -m elastic.cli.stage_opt          --help  # HF checkpoint → packed NVMe blob + manifest
    python -m elastic.cost.cli_measure_units --help  # unit costs → units_<model>_S<S>.json
    python -m elastic.policy.run_solve       --help  # units + (N_seq, S, gen) cell → alpha_curve.csv
    python -m elastic.cli.run_bench          --help  # one cell end to end → throughput

Staging produces the blob a run reads, measuring produces the units record
the solver consumes, and the solver produces the per-layer α curve the bench
takes. There is no central subcommand router — each CLI module owns its own
argparse `main()` and is invoked directly.
"""

import sys


def main() -> None:
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    main()
