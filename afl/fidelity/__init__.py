"""⚑ Built before the generator, on purpose.

If the harness arrives after the thing it judges, the thresholds get chosen to fit the results.
Three levels of evidence, one gate: level 3 (utility) is the bar; levels 1 and 2 explain it.

The bars themselves live in `config/fidelity/thresholds.yaml`, and `provenance` is what checks
they predate the numbers they judge rather than asserting it.
"""

from afl.fidelity.provenance import ThresholdError, ThresholdProvenance
from afl.fidelity.scorecard import Scorecard, Thresholds, build

__all__ = ["Scorecard", "ThresholdError", "ThresholdProvenance", "Thresholds", "build"]
