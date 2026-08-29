"""Does the wording still say only what the artefacts support?

    python scripts/check_guardrails.py            # every guardrail, then anything it caught
    python scripts/check_guardrails.py --quiet    # violations only, for a hook or CI

`check_claims.py` proves the numbers are the artefacts'. This proves the sentences around them are
too. Reads `docs/guardrails.yaml`, walks every document in scope one sentence at a time, and fails
on any sentence that overstates what was measured — or that names an instrument without saying
what it is worth.

Exit code 0 means the write-up claims what the runs support. Anything else is an overstatement,
which in a submission is the expensive kind of defect: a reviewer who finds one stops trusting the
rest of the document, including the parts that were right.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afl.repro.guardrails import RULES_PATH, Guardrails, report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rules", type=Path, default=RULES_PATH)
    p.add_argument("--root", type=Path, default=Path("."), help="repository root to check")
    p.add_argument("--quiet", action="store_true", help="print only what failed")
    p.add_argument("--list-documents", action="store_true", help="print the documents in scope")
    args = p.parse_args()

    rails = Guardrails.load(args.rules)
    documents = rails.files(args.root)
    if args.list_documents:
        for path in documents:
            print(path)
        return 0

    violations = rails.audit(args.root)
    unenforced = rails.unenforced()

    if args.quiet:
        for violation in violations:
            print(violation.line())
    else:
        print(report(rails, violations, len(documents)))

    if unenforced:
        print(
            "\nFAIL: "
            + ", ".join(g.id for g in unenforced)
            + " — listed with no rule behind it. A guardrail nobody can fail is decoration."
        )
        return 1
    if violations:
        print(
            f"\nFAIL: {len(violations)} sentence(s) claim more than the artefacts support.\n"
            f"Reword the sentence, or — if the claim is now earned — say which run earned it and "
            f"add the exception to {args.rules}."
        )
        return 1
    print(
        f"\nOK: {len(rails.guardrails)} guardrails enforced over {len(documents)} documents, "
        f"no sentence overstates the runs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
