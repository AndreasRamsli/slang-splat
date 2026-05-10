from __future__ import annotations

import numpy as np

from torch_examples.train_colmap_garden_torch import summarize_trace_substeps


def test_summarize_trace_substeps_computes_ms_and_percentages() -> None:
    trace = {
        "traceEvents": [
            {"name": "train/data_loader", "ph": "X", "dur": 1_000.0},
            {"name": "train/forward", "ph": "X", "dur": 3_000.0},
            {"name": "train/backward", "ph": "X", "dur": 6_000.0},
            {"name": "ignored", "ph": "X", "dur": 9_999.0},
            {"name": "train/forward", "ph": "i", "dur": 5_000.0},
        ]
    }

    summary = summarize_trace_substeps(trace)

    assert np.isclose(summary["total_ms"], 10.0)
    assert np.isclose(summary["data_loader_ms"], 1.0)
    assert np.isclose(summary["forward_ms"], 3.0)
    assert np.isclose(summary["backward_ms"], 6.0)
    assert np.isclose(summary["data_loader_pct"], 10.0)
    assert np.isclose(summary["forward_pct"], 30.0)
    assert np.isclose(summary["backward_pct"], 60.0)