"""Does every quoted number still fall out of the artefact it names?

    python scripts/check_claims.py            # the registry, then the covered sections
    python scripts/check_claims.py --quiet    # failures only, for a hook or CI

Reads `docs/claims.yaml`, recomputes every claim over the committed artefacts, and fails unless
each one formats to exactly the string the documents quote — and unless every number inside a
covered section is a registered claim or an allowed constant.

Exit code 0 means the documents and the artefacts agree. Anything else is a stale number, which
in this repository is a defect and not a rounding difference: `artifacts/abcd/README.md` records
what one of those cost the last time it went unnoticed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afl.repro.claims import REGISTRY_PATH, Registry, report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    p.add_argument("--root", type=Path, default=Path("."), help="repository root to check against")
    p.add_argument("--quiet", action="store_true", help="print only what failed")
    args = p.parse_args()

    registry = Registry.load(args.registry)
    checks = registry.check(args.root)
    gaps = registry.coverage(args.root)

    failed = [c for c in checks if not c.ok]
    if args.quiet:
        for check in failed:
            print(check.line())
        for gap in gaps:
            print(gap.line())
    else:
        print(report(checks, gaps))

    if failed or gaps:
        print(
            f"\nFAIL: {len(failed)} claim(s) stale, {len(gaps)} number(s) with no artefact behind "
            f"them.\nFix the document, fix the registry, or regenerate the artefact — but one of "
            f"the three is wrong."
        )
        return 1
    print(f"\nOK: {len(checks)} claims verified against {len(registry.artifacts)} artefacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
