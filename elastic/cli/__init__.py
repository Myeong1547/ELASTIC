"""elastic CLI modules — each exposes its own argparse `main()` and is
invoked directly (there is no central subcommand router; `python -m elastic`
lists the whole pipeline):

    python -m elastic.cli.stage_opt   # HF checkpoint → packed NVMe blob + manifest
    python -m elastic.cli.run_bench   # end-to-end throughput bench for one cell
"""
