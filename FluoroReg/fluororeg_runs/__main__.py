"""Command line: python -m fluororeg_runs check <run_dir> [<run_dir> ...] [--json FILE]."""
from __future__ import annotations

import argparse
import json
import sys

from .contract import contract_check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fluororeg_runs",
                                     description="Tools for xreg run directories (run_manifest.json schema 4).")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="contract check of one or more run directories")
    check.add_argument("run_dirs", nargs="+", help="run directories (or run_manifest.json files)")
    check.add_argument("--tol-px", type=float, default=0.5, help="landmark residual tolerance in pixels")
    check.add_argument("--joint-tol-mm", type=float, default=1e-3, help="T_i = J F_i translation tolerance")
    check.add_argument("--joint-tol-rot", type=float, default=1e-5,
                       help="T_i = J F_i tolerance on rotation matrix entries")
    check.add_argument("--json", metavar="FILE", help="also write the reports as JSON")
    args = parser.parse_args(argv)

    reports = [contract_check(d, args.tol_px, args.joint_tol_mm, args.joint_tol_rot)
               for d in args.run_dirs]
    for rep in reports:
        print(rep.format())
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in reports], f, indent=1)
    failed = [r.run_dir for r in reports if not r.ok]
    print(f"{len(reports) - len(failed)} of {len(reports)} run(s) pass the contract check")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
