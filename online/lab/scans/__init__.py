"""Each module implements one paper's detection idea as a standalone scan.

Every scan exposes `run(ctx, prompts) -> ScanResult` so `run_scan.py` can call
them uniformly and the user can run one idea or all of them in a single run.
"""
