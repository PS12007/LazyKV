"""Collect the machine description that every result is conditional on."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Any


def _powershell_json(script: str) -> Any:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", f"{script} | ConvertTo-Json -Depth 4"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return json.loads(out.stdout) if out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _nvidia_smi(fields: list[str]) -> dict[str, str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    values = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
    return dict(zip(fields, values))


def windows_host() -> dict[str, Any]:
    cpu = _powershell_json(
        "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed"
    )
    dimms = _as_list(
        _powershell_json(
            "Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed,ConfiguredClockSpeed,Manufacturer,PartNumber,DeviceLocator"
        )
    )
    os_info = _powershell_json("Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber")
    disks = _as_list(
        _powershell_json("Get-PhysicalDisk | Select-Object FriendlyName,BusType,MediaType,Size")
    )
    battery = _as_list(_powershell_json("Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining"))
    # HwSchMode: 2 = hardware-accelerated GPU scheduling on, 1 = off. It changes how WDDM
    # queues our copies and kernels, so it is part of the experimental condition.
    hags = _powershell_json(
        "(Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers' -Name HwSchMode -ErrorAction SilentlyContinue).HwSchMode"
    )
    try:
        power_plan = subprocess.run(
            ["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        power_plan = None
    return {
        "os": os_info,
        "cpu": cpu,
        "ram_dimms": dimms,
        "ram_total_bytes": sum(int(d.get("Capacity") or 0) for d in dimms),
        # One DIMM means single-channel memory: halves host memory bandwidth, which bounds
        # both pageable copies and any CPU-side attention (design option (c)).
        "ram_dimm_count": len(dimms),
        "disks": disks,
        # BatteryStatus 2 = on AC. A laptop on battery is a different machine.
        "battery": battery,
        "hags_hwschmode": hags,
        "power_plan": power_plan,
    }


def gpu_and_software() -> dict[str, Any]:
    import numpy
    import torch
    import transformers

    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "torch_num_threads_default": torch.get_num_threads(),
        "env": {
            k: os.environ.get(k)
            for k in ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "nvidia_smi": _nvidia_smi(
            [
                "name",
                "driver_version",
                "memory.total",
                "pcie.link.gen.max",
                "pcie.link.gen.hostmax",
                "pcie.link.width.max",
                "power.limit",
                "power.max_limit",
                "vbios_version",
                "driver_model.current",
            ]
        ),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        fields: dict[str, Any] = {}
        for name in dir(props):
            if name.startswith("_"):
                continue
            value = getattr(props, name)
            if callable(value):
                continue
            fields[name] = value if isinstance(value, (int, float, str, bool)) or value is None else str(value)
        info["cuda_device_properties"] = fields
        info["cuda_device_properties_repr"] = repr(props)
        info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
        info["cuda_arch_list"] = torch.cuda.get_arch_list()
        free, total = torch.cuda.mem_get_info(0)
        info["cuda_mem_free_bytes_at_start"] = free
        info["cuda_mem_total_bytes"] = total
    return info


def expandable_segments_probe() -> dict[str, Any]:
    """Check whether PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is honored here.

    Run in a subprocess because the allocator config is read once at CUDA init.
    """
    code = (
        "import warnings, torch;"
        "warnings.simplefilter('always');"
        "import io, contextlib;"
        "x = torch.empty(64 * 2**20, dtype=torch.uint8, device='cuda');"
        "torch.cuda.synchronize();"
        "print('ALLOC_OK')"
    )
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ran": False, "error": str(exc)}
    stderr = out.stderr.strip()
    return {
        "ran": True,
        "alloc_ok": "ALLOC_OK" in out.stdout,
        "warned_not_supported": "not supported" in stderr.lower(),
        "stderr_tail": stderr[-500:],
    }


def collect() -> dict[str, Any]:
    return {
        "host": windows_host() if platform.system() == "Windows" else {"platform": platform.platform()},
        "software_gpu": gpu_and_software(),
        "expandable_segments": expandable_segments_probe(),
    }
