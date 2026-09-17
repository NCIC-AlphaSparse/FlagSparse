# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared benchmark timing helpers for test scripts."""

import time


def filtered_avg_ms(times, tolerance=0.10):
    """Return a median-window filtered average in milliseconds.

    This matches the filtering policy used by the existing SpSM/SpSV scripts:
    compute the median, keep samples within median +/- 10% by default, and
    average the kept samples. If all samples are filtered, fall back to median.
    """

    if not times:
        return None
    values = [float(t) for t in times]
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    n = len(ordered)
    if n % 2 == 0:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    else:
        median = ordered[n // 2]
    if median == 0.0:
        kept = [t for t in ordered if t == 0.0]
    else:
        delta = abs(median) * float(tolerance)
        lo = median - delta
        hi = median + delta
        kept = [t for t in ordered if lo <= t <= hi]
    return sum(kept) / len(kept) if kept else median


def cuda_event_benchmark_filtered(op, warmup, iters):
    import torch

    out = None
    for _ in range(max(0, int(warmup))):
        out = op()
    torch.cuda.synchronize()
    samples = []
    for _ in range(max(1, int(iters))):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = op()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return out, filtered_avg_ms(samples)


def cupy_event_benchmark_filtered(op, warmup, iters):
    import cupy as cp

    out = None
    for _ in range(max(0, int(warmup))):
        out = op()
    cp.cuda.runtime.deviceSynchronize()
    samples = []
    for _ in range(max(1, int(iters))):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        out = op()
        end.record()
        end.synchronize()
        samples.append(cp.cuda.get_elapsed_time(start, end))
    return out, filtered_avg_ms(samples)


def cpu_wall_benchmark_filtered(op, warmup, iters):
    out = None
    for _ in range(max(0, int(warmup))):
        out = op()
    samples = []
    for _ in range(max(1, int(iters))):
        start = time.perf_counter()
        out = op()
        samples.append((time.perf_counter() - start) * 1000.0)
    return out, filtered_avg_ms(samples)
