"""Resumable, append-only result storage with reproducibility metadata.

Each experiment writes one JSON-lines file per seed under ``<output>/raw/``.
Every record carries a ``key`` string that identifies its configuration, so
a re-run skips configurations already present and only computes the missing
ones. ``metadata.json`` records the command line, configuration, software
versions and platform of the most recent invocation (earlier invocations are
kept in a list so partial runs remain auditable).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path


def environment_metadata() -> dict:
    import matplotlib
    import numpy
    import pandas
    import scipy
    import sklearn

    from . import __version__

    return {
        "predictive_abstraction": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


class ResultStore:
    def __init__(self, output: Path, name: str):
        self.output = Path(output)
        self.raw_dir = self.output / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.name = name

    def path_for(self, seed: int) -> Path:
        return self.raw_dir / f"{self.name}-seed-{seed}.jsonl"

    def load(self, seed: int) -> dict[str, dict]:
        path = self.path_for(seed)
        records: dict[str, dict] = {}
        if path.is_file():
            with path.open() as stream:
                for line in stream:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        records[record["key"]] = record
        return records

    def append(self, seed: int, record: dict) -> None:
        clean = {k: v for k, v in record.items() if not k.startswith("_")}
        with self.path_for(seed).open("a") as stream:
            stream.write(json.dumps(clean) + "\n")

    def load_all(self) -> list[dict]:
        records = []
        for path in sorted(self.raw_dir.glob(f"{self.name}-seed-*.jsonl")):
            with path.open() as stream:
                records.extend(json.loads(line) for line in stream if line.strip())
        return records

    def check_consistent(self, config: dict, keys: tuple[str, ...]) -> None:
        """Refuse to resume into a directory whose earlier runs used different data settings.

        Records are keyed by configuration, but the simulated episodes depend
        on the training-pool size and the split sizes. Mixing them would
        silently combine results computed on different data.
        """
        path = self.output / "metadata.json"
        if not path.is_file():
            return
        previous = json.loads(path.read_text()).get("latest", {}).get("config", {})
        for key in keys:
            if key in previous and previous[key] != config[key]:
                raise SystemExit(
                    f"{path} was produced with {key}={previous[key]!r} but this run uses "
                    f"{config[key]!r}; use a new --output directory")

    def write_metadata(self, config: dict, *, extra: dict | None = None) -> None:
        path = self.output / "metadata.json"
        history = []
        if path.is_file():
            try:
                history = json.loads(path.read_text()).get("invocations", [])
            except json.JSONDecodeError:
                history = []
        invocation = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": " ".join(sys.argv),
            "config": config,
            "environment": environment_metadata(),
        }
        if extra:
            invocation.update(extra)
        history.append(invocation)
        path.write_text(json.dumps({
            "experiment": self.name,
            "latest": invocation,
            "invocations": history,
        }, indent=2) + "\n")
