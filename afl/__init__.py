"""Adaptive Fraud Simulation Lab.

Two halves that never import each other:
  afl.attack  (red side)  ─┐
                           ├─> afl.contract  <─ the only shared surface
  afl.defend  (blue side) ─┘

afl.loop.closed_loop is where they meet, and it only speaks contract types.
"""

__version__ = "0.1.0"
