"""Exact input identity resolution in the process that will open PortAudio."""


def input_host_api(sd, device):
    index = device.get("hostapi")
    if not isinstance(index, int):
        return None
    name = sd.query_hostapis(index).get("name")
    return name if isinstance(name, str) else None


def resolve_input_device(device=None, *, name=None, host_api=None, sd=None):
    """Resolve an exact name/API pair; never replace a missing choice by default.

    Integer-only callers retain their historical process-local index semantics.
    ``None`` without an identity means the caller intentionally chose automatic.
    """
    if name is None and isinstance(device, str):
        name, device = device, None
    if name is None and host_api is not None:
        raise ValueError("Input host API requires a device name")
    if host_api is not None and (not isinstance(host_api, str) or not host_api or len(host_api) > 512):
        raise ValueError("Invalid input host API")
    if name is None and device is None:
        return None
    if sd is None:
        import sounddevice as sd
    devices = list(sd.query_devices())
    if name is not None:
        if not isinstance(name, str) or not name or len(name) > 512:
            raise ValueError("Invalid input device name")
        matches = [
            (i, row)
            for i, row in enumerate(devices)
            if row.get("name") == name
            and row.get("max_input_channels", 0) > 0
            and (host_api is None or input_host_api(sd, row) == host_api)
        ]
        if not matches:
            raise ValueError(f"Selected input device is unavailable: {name}")
        if len(matches) != 1:
            raise ValueError(f"Selected input device is ambiguous: {name}")
        index, selected = matches[0]
    else:
        if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device < len(devices):
            raise ValueError("Invalid input device index")
        index, selected = device, devices[device]
        if selected.get("max_input_channels", 0) <= 0:
            raise ValueError("Selected device has no input channels")
    return {"index": index, "name": selected["name"], "host_api": input_host_api(sd, selected)}
