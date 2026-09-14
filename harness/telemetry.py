"""Background GPU telemetry via a single long-lived ``nvidia-smi -lms`` process.

Why nvidia-smi and not NVML bindings: pynvml is not in the locked env and adding it
needs sign-off. One streaming process costs far less than spawning nvidia-smi per sample.
Why log at all: this is a laptop. Clock, temperature, power and PCIe link state move
during a benchmark, and a result without the trace cannot be told apart from a thermal
artifact. PCIe link gen matters too: laptop GPUs downshift the link at idle.
"""

from __future__ import annotations

import csv
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

log = logging.getLogger(__name__)

FIELDS: tuple[str, ...] = (
    "temperature.gpu",
    "power.draw",
    "clocks.current.graphics",
    "clocks.current.memory",
    "pstate",
    "utilization.gpu",
    "memory.used",
    "pcie.link.gen.current",
    "pcie.link.width.current",
    "clocks_event_reasons.active",
)


@dataclass
class TelemetryLogger:
    csv_path: Path
    interval_ms: int = 500
    _rows: list[dict[str, str | float]] = field(default_factory=list)
    _marks: list[dict[str, str | float]] = field(default_factory=list)
    _proc: subprocess.Popen[str] | None = None
    _thread: threading.Thread | None = None
    _t0: float = 0.0

    def __enter__(self) -> TelemetryLogger:
        self._t0 = time.perf_counter()
        cmd = [
            "nvidia-smi",
            f"--query-gpu={','.join(FIELDS)}",
            "--format=csv,noheader,nounits",
            f"-lms={self.interval_ms}",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
            )
        except FileNotFoundError:
            log.warning("nvidia-smi not found; telemetry disabled")
            return self
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != len(FIELDS):
                continue
            row: dict[str, str | float] = {"t_s": time.perf_counter() - self._t0}
            row.update(dict(zip(FIELDS, parts)))
            self._rows.append(row)

    def mark(self, label: str) -> None:
        """Record a section boundary so the trace can be annotated in plots."""
        self._marks.append({"t_s": time.perf_counter() - self._t0, "label": label})

    @property
    def marks(self) -> list[dict[str, str | float]]:
        return list(self._marks)

    def summary(self) -> dict[str, object]:
        def numeric(key: str) -> list[float]:
            out: list[float] = []
            for r in self._rows:
                try:
                    out.append(float(r[key]))
                except (ValueError, KeyError):
                    pass
            return out

        temps = numeric("temperature.gpu")
        power = numeric("power.draw")
        clocks = numeric("clocks.current.graphics")
        gens = numeric("pcie.link.gen.current")
        widths = numeric("pcie.link.width.current")
        active_reasons = {str(r.get("clocks_event_reasons.active")) for r in self._rows}
        return {
            "samples": len(self._rows),
            "interval_ms": self.interval_ms,
            "temperature_c": {"min": min(temps), "max": max(temps)} if temps else None,
            "power_w": {"min": min(power), "max": max(power)} if power else None,
            "graphics_clock_mhz": {"min": min(clocks), "max": max(clocks)} if clocks else None,
            "pcie_link_gen_seen": sorted({int(g) for g in gens}),
            "pcie_link_width_seen": sorted({int(w) for w in widths}),
            "clock_event_reason_bitmasks_seen": sorted(active_reasons),
        }

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["t_s", *FIELDS])
            writer.writeheader()
            writer.writerows(self._rows)
