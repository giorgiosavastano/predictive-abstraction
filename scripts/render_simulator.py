"""Render the essay's simulator walkthrough from one reproducible trajectory.

Run from the repository root with ``uv run python scripts/render_simulator.py``.
This writes only docs/figures/sorting-experiment.svg; no research result changes.
"""
from __future__ import annotations

from copy import deepcopy
from html import escape
from pathlib import Path

import numpy as np

from macro_prediction.features import global_features, macro_features
from macro_prediction.simulator import N, _sweep, make_episodes


def render() -> str:
    arrays, probabilities = make_episodes(1, np.random.default_rng(314159))
    start = arrays.copy()
    rng = np.random.default_rng(20261108)

    # Replay the actual first parity phase for the close-up. Verify the manual
    # trace against the simulator's complete first sweep, including RNG state.
    replay = np.random.default_rng()
    replay.bit_generator.state = deepcopy(rng.bit_generator.state)
    first_phase = start.copy()
    left = np.arange(0, N - 1, 2)
    trials = replay.random((1, len(left)))
    inverted = start[0, left] > start[0, left + 1]
    swapped = inverted & (trials[0] < probabilities[0, left])
    succeeded = int(left[swapped][0])
    failed = int(left[inverted & ~swapped][0])
    for edge in left[swapped]:
        first_phase[0, edge:edge + 2] = first_phase[0, edge:edge + 2][::-1]
    manual = first_phase.copy()
    odd = np.arange(1, N - 1, 2)
    odd_trials = replay.random((1, len(odd)))
    odd_swaps = (manual[0, odd] > manual[0, odd + 1]) & (odd_trials[0] < probabilities[0, odd])
    for edge in odd[odd_swaps]:
        manual[0, edge:edge + 2] = manual[0, edge:edge + 2][::-1]

    states = {0: start.copy()}
    for sweep in range(1, 13):
        _sweep(arrays, probabilities, rng)
        if sweep == 1:
            np.testing.assert_array_equal(arrays, manual)
            assert rng.bit_generator.state == replay.bit_generator.state
        if sweep in (1, 4, 12):
            states[sweep] = arrays.copy()
    disorder = lambda a: float(np.abs(a[0] - np.arange(N)).mean())
    assert disorder(states[12]) <= disorder(states[4]) <= disorder(states[1]) <= disorder(states[0])
    assert all(np.array_equal(np.sort(a[0]), np.arange(N)) for a in states.values())
    assert macro_features(start, probabilities).shape == (1, 27)

    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 1110" role="img" aria-labelledby="title desc">
<title id="title">From individual numbers to a supplied summary</title>
<desc id="desc">A real 64-element trajectory from the stochastic sorting simulator. A disturbed array recovers through local compare-and-swap operations. One successful and one failed attempt are shown from the first even-edge phase. Full states retain element values and exchange probabilities; the supplied summary retains three global and twenty-four regional numbers.</desc>
<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172b3a;font-size:22px}.heading{font-size:26px;font-weight:700}.small{font-size:19px}.bar{fill:#26678e}.box{fill:#f3f6f8;stroke:#9aaab5;stroke-width:1.5}</style>
<defs><pattern id="fault" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M0 8L8 0" stroke="#9b681d" stroke-opacity=".35"/></pattern></defs>
<rect width="720" height="1110" fill="white"/>
''']
    def text(x, y, s, cls=''):
        parts.append(f'<text x="{x}" y="{y}" class="{cls}">{escape(s)}</text>')
    text(24, 36, 'One array, two descriptions', 'heading')
    text(24, 66, '64 numbers · local comparisons · unreliable swaps')
    fault_edges = np.flatnonzero(probabilities[0, :-1] < .89)
    fault_start, fault_end = int(fault_edges[0]), int(fault_edges[-1] + 1)
    x0, cell_width = 24, 10.5

    def row(a, top, label, mark_fault=True):
        text(24, top, label)
        base = top + 83
        if mark_fault:
            fx = x0 + fault_start * cell_width
            fw = (fault_end - fault_start + 1) * cell_width
            parts.append(f'<rect x="{fx:.1f}" y="{top+10}" width="{fw:.1f}" height="76" fill="url(#fault)"/>')
        for i, value in enumerate(a[0]):
            height = 3 + 70 * int(value) / 63
            parts.append(f'<rect class="bar" x="{x0+i*cell_width:.1f}" y="{base-height:.2f}" width="8.5" height="{height:.2f}"/>')
        parts.append(f'<path d="M24 {base+2}H696" stroke="#536674"/>')

    row(np.arange(N)[None, :], 108, 'Ordered: each number is in its own position', False)
    row(states[0], 226, f'Disturbed start · disorder {disorder(states[0]):.2f}')
    text(24, 339, f'Hatched interval: less reliable edges {fault_start}–{fault_end-1}', 'small')
    text(24, 382, 'Close-up: two attempts in the first half-round', 'heading')
    for x, edge, label in [(24, succeeded, 'Swap succeeds'), (380, failed, 'Swap fails')]:
        parts.append(f'<rect class="box" x="{x}" y="399" width="316" height="120" rx="6"/>')
        text(x+14, 428, f'{label} · edge {edge}')
        a, b = map(int, start[0, edge:edge+2])
        c, d = map(int, first_phase[0, edge:edge+2])
        text(x+14, 466, f'{a}, {b}  →  {c}, {d}')
        text(x+14, 500, f'Success probability {probabilities[0, edge]:.2f}', 'small')
    row(states[1], 556, f'After 1 full round · disorder {disorder(states[1]):.2f}')
    row(states[4], 668, f'After 4 rounds · disorder {disorder(states[4]):.2f}')
    row(states[12], 780, f'After 12 rounds · disorder {disorder(states[12]):.2f}')
    text(24, 910, 'What the predictor receives at the start', 'heading')
    parts.append('<rect class="box" x="24" y="928" width="672" height="68" rx="6"/>')
    text(38, 956, 'Full state: 64 values + 64 reliability entries')
    text(38, 982, '128 numbers, including one padding entry', 'small')
    parts.append('<rect class="box" x="24" y="1008" width="672" height="78" rx="6"/>')
    text(38, 1037, 'Supplied summary: 3 global + 3 × 8 regional = 27')
    g = global_features(start, probabilities)[0]
    text(38, 1068, f'Global values (normalised): {g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}', 'small')
    parts.append('</svg>\n')
    return '\n'.join(parts)


if __name__ == '__main__':
    destination = Path(__file__).resolve().parents[1] / 'docs/figures/sorting-experiment.svg'
    destination.write_text(render())
    print(destination)
