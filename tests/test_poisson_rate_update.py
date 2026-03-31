"""
tests/test_poisson_rate_update.py

Question: which method updates PoissonGroup.rates between net.run() calls
WITHOUT triggering Brian2 C++ recompilation?

Approaches tested:
  A) group.rates = rate * Hz          -- property setter (current code)
  B) group.rates[:] = rate * Hz       -- in-place slice assignment
  C) group.rates_ = rate              -- trailing-underscore raw SI array (1/s)

Diagnostic signals:
  - time per step: if step 1+ is as fast as step 0 post-compilation -> no recompile
  - Brian2 cache file count: new .pyd/.so file between steps = recompile happened

Run with numpy backend (fast, no compiler needed):
    wenv310\Scripts\python.exe tests\test_poisson_rate_update.py

Run with Cython/C++ backend (tests the real production backend):
    wenv310\Scripts\python.exe tests\test_poisson_rate_update.py --cython

Results with numpy backend confirmed: all three approaches are stable (no recompile).
The --cython flag tests whether the same holds for the C++ backend.
"""

import sys, time, os, glob
import numpy as np

use_cython = "--cython" in sys.argv

from brian2 import prefs
if use_cython:
    print("Backend: Cython/C++ (will trigger real compilation on first run)")
else:
    prefs.codegen.target = "numpy"
    print("Backend: numpy (fast, no compiler needed)")

from brian2 import (
    NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,
    mV, ms, Hz, second, start_scope,
)

RATES  = [50.0, 100.0, 150.0, 80.0, 120.0]   # Hz values to cycle through
N_ASC  = 10
N_BODY = 10
DT_RUN = 10 * ms


def _cache_count():
    """Count compiled extension files in the Brian2 cython cache."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cython", "brian_extensions")
    if not os.path.isdir(cache_dir):
        return 0
    return len(
        glob.glob(os.path.join(cache_dir, "*.pyd")) +
        glob.glob(os.path.join(cache_dir, "*.so"))
    )


def run_approach(label, setup_fn, update_fn):
    start_scope()

    neu = NeuronGroup(
        N_BODY,
        "dv/dt = -v/(10*ms) : volt",
        threshold="v > -45*mV",
        reset="v = -52*mV",
        refractory=2 * ms,
        method="euler",
    )
    neu.v = -52 * mV

    asc_group, extra_objects = setup_fn()

    syn = Synapses(asc_group, neu, "w : volt", on_pre="v += w")
    syn.connect(i=list(range(N_ASC)), j=list(range(N_BODY)))
    syn.w = 5 * mV

    mon = SpikeMonitor(neu)
    net = Network(neu, syn, mon, asc_group, *extra_objects)

    print(f"\n-- {label} --")
    times = []
    cache_before = _cache_count()

    for i, rate in enumerate(RATES):
        t0 = time.time()
        update_fn(asc_group, rate)
        net.run(DT_RUN)
        elapsed = (time.time() - t0) * 1000
        times.append(elapsed)

        cache_after = _cache_count()
        new_files = cache_after - cache_before
        cache_before = cache_after

        recompile_tag = (
            f"  *** +{new_files} cache file(s) — RECOMPILED" if new_files > 0 else ""
        )
        print(
            f"  step {i}  rate={rate:.0f}Hz  time={elapsed:.1f}ms{recompile_tag}"
        )

    # Heuristic: if any step after the first is > 3x the median of later steps, flag it
    if len(times) > 1:
        later = times[1:]
        median_later = sorted(later)[len(later) // 2]
        slow_steps = [i for i, t in enumerate(times) if t > max(median_later * 3, 50)]
        if slow_steps and slow_steps != [0]:
            print(f"  WARNING WARNING: slow steps detected at indices {slow_steps} — recompilation likely")
        else:
            print(f"  OK Times stable after step 0 — no per-step recompilation detected")

    return times


# -- Run all three approaches ------------------------------------------------

times_a = run_approach(
    "A: g.rates = rate * Hz  (property setter — current code)",
    setup_fn=lambda: (PoissonGroup(N=N_ASC, rates=50 * Hz), []),
    update_fn=lambda g, r: setattr(g, "rates", r * Hz),
)

times_b = run_approach(
    "B: g.rates[:] = rate * Hz  (in-place slice assignment)",
    setup_fn=lambda: (PoissonGroup(N=N_ASC, rates=50 * Hz), []),
    update_fn=lambda g, r: g.rates.__setitem__(slice(None), r * Hz),
)

times_c = run_approach(
    "C: g.rates_ = rate  (trailing-underscore raw SI array, 1/s)",
    setup_fn=lambda: (PoissonGroup(N=N_ASC, rates=50 * Hz), []),
    update_fn=lambda g, r: setattr(g, "rates_", r),
)

print("\n-- Summary --")
for label, times in [("A (setter)", times_a), ("B (slice)", times_b), ("C (rates_)", times_c)]:
    print(f"  {label}: {[f'{t:.0f}' for t in times]} ms")

print("\nDone.")
