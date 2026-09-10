"""Pinned native runtime installation and a session-owned llama-server.

The server inherits its pipeline process group. The pipeline supervisor may
kill that group; normal teardown terminates only the exact child we created.
No existing listener is adopted or stopped.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import platform
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

from engines.model_paths import default_models_dir, load_model_manifest, resolve_model_path

REVISION = "91f6a6cf361385700bbe15981f0f39909df77498"
BUILD = "b10883"
# GitHub release asset digests, read from the upstream API at implementation.
ASSETS = {
    "Darwin-arm64-cpu": (
        "llama-b10883-bin-macos-arm64.tar.gz",
        "a83a885bf2fa4ffb7c11b3c8c6ed7e7ff8f7bd61733b1d4fa2ce8ec7cb587588",
    ),
    "Darwin-x86_64-cpu": (
        "llama-b10883-bin-macos-x64.tar.gz",
        "7983c8a7ad4c67595c390c612c67fe7cb367517e86d29e00f96138b65ecd7a23",
    ),
    "Linux-x86_64-cpu": (
        "llama-b10883-bin-ubuntu-x64.tar.gz",
        "bb7df4783b12f164a1e829d4e692c216324ad919883f9e1c8abc77262db29145",
    ),
    "Linux-arm64-cpu": (
        "llama-b10883-bin-ubuntu-arm64.tar.gz",
        "2548d08f97c81c8b761614132db9d355cf59b9ce0feaa91a57ccc345924d4858",
    ),
    "Windows-x86_64-cpu": (
        "llama-b10883-bin-win-cpu-x64.zip",
        "05865586d8cc235e7ed1e975e0df4c6d02ba81cab637845aed888ad5fe4ba061",
    ),
    "Windows-x86_64-cuda": (
        "llama-b10883-bin-win-cuda-12.4-x64.zip",
        "ec27d173749dba3b03080b4580070d8b07c6108e4972e39861d9ea63638177e2",
    ),
}
CUDA_RUNTIME = (
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def platform_key(backend: str) -> str:
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return f"{platform.system()}-{machine}-{backend}"


def native_root(models_dir: Path | None = None) -> Path:
    return (models_dir or default_models_dir()) / "native" / BUILD


def resolve_native(backend: str, models_dir: Path | None = None, *, verify: bool = True) -> tuple[Path, dict]:
    root = native_root(models_dir) / platform_key(backend)
    try:
        data = json.loads((root / "installed.json").read_text())
        executable = (root / data["executable"]).resolve()
        if (
            data["revision"] != REVISION
            or data["platform"] != platform_key(backend)
            or not executable.is_relative_to(root.resolve())
        ):
            raise ValueError("native runtime identity mismatch")
        for relative, digest in data["files"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or (verify and sha256(path) != digest):
                raise ValueError(f"native runtime file changed: {relative}")
        if data["executable"] not in data["files"] or not executable.is_file():
            raise ValueError("native executable missing from inventory")
        return executable, data
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FileNotFoundError(
            f"Verified {backend} llama-server missing; run setup for the selected lite profile ({exc})"
        ) from exc


def _extract(archive: Path, destination: Path):
    """Accept only confined regular archive entries; tar data filter validates links."""
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = (destination / member.filename).resolve()
                if (
                    not target.is_relative_to(destination.resolve())
                    or (member.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("Unsafe native ZIP member")
            for member in z.infolist():
                target = destination / member.filename
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
    else:
        with tarfile.open(archive) as t:
            # The archive digest is pinned; the data filter rejects escaping
            # paths and links before any member is extracted.
            t.extractall(destination, filter="data")


def install_native(backend: str, models_dir: Path | None = None, *, offline: bool = False, build: bool = False) -> Path:
    try:
        return resolve_native(backend, models_dir)[0]
    except FileNotFoundError:
        pass
    key = platform_key(backend)
    parent = native_root(models_dir)
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / key
    if final.exists():
        raise ValueError(f"Existing native artifact is invalid: {final}; move it aside before setup")
    stage = Path(tempfile.mkdtemp(prefix=".native-", dir=parent))
    try:
        if key == "Linux-x86_64-cuda":
            if not build:
                raise ValueError(
                    "Linux CUDA requires a pinned sm_75 build: rerun setup --build-native with git, CMake and CUDA 12 installed"
                )
            if offline:
                raise ValueError("Offline native setup needs a previously prepared verified native cache")
            source = stage / "source"
            subprocess.run(["git", "init", str(source)], check=True)
            subprocess.run(
                ["git", "-C", str(source), "fetch", "--depth=1", "https://github.com/ggml-org/llama.cpp.git", REVISION],
                check=True,
            )
            subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
            actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            if actual != REVISION:
                raise ValueError("llama.cpp source revision mismatch")
            subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(source),
                    "-B",
                    str(source / "build"),
                    "-DGGML_CUDA=ON",
                    "-DCMAKE_CUDA_ARCHITECTURES=75",
                    "-DLLAMA_CURL=OFF",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                check=True,
            )
            subprocess.run(
                ["cmake", "--build", str(source / "build"), "--target", "llama-server", "-j", "3"], check=True
            )
            shutil.copytree(source / "build" / "bin", stage / "bin", symlinks=False)
            shutil.rmtree(source)
        else:
            if key not in ASSETS:
                raise ValueError(f"No pinned native artifact for {key}")
            archive_dir = parent / "archives"
            archive_dir.mkdir(exist_ok=True)
            for filename, digest in [ASSETS[key]] + ([CUDA_RUNTIME] if key.endswith("cuda") else []):
                archive = archive_dir / filename
                if not archive.exists():
                    if offline:
                        raise FileNotFoundError(f"Offline native archive missing: {archive}")
                    from operator_app.setup import _download_direct

                    _download_direct(
                        f"https://github.com/ggml-org/llama.cpp/releases/download/{BUILD}/{filename}", archive, None
                    )
                if sha256(archive) != digest:
                    raise ValueError(f"Native archive checksum mismatch: {archive}")
                _extract(archive, stage)
        candidates = list(stage.rglob("llama-server.exe" if platform.system() == "Windows" else "llama-server"))
        if len(candidates) != 1:
            raise ValueError("Native archive must contain exactly one llama-server executable")
        executable = candidates[0]
        # Upstream Windows CUDA runtime ZIP may put DLLs at a different depth
        # from llama-server.exe. Keep the exact files beside the executable.
        if platform.system() == "Windows":
            for dll in list(stage.rglob("*.dll")):
                target = executable.parent / dll.name
                if dll != target:
                    if target.exists() and sha256(target) != sha256(dll):
                        raise ValueError(f"Conflicting native DLL: {dll.name}")
                    if not target.exists():
                        shutil.copy2(dll, target)
        if platform.system() != "Windows":
            executable.chmod(executable.stat().st_mode | 0o111)
        files = {str(p.relative_to(stage)): sha256(p) for p in stage.rglob("*") if p.is_file()}
        data = {
            "revision": REVISION,
            "build": BUILD,
            "platform": key,
            "cuda_architectures": [75] if key == "Linux-x86_64-cuda" else None,
            "executable": str(executable.relative_to(stage)),
            "files": files,
        }
        (stage / "installed.json").write_text(json.dumps(data, indent=2) + "\n")
        stage.rename(final)
        return resolve_native(backend, models_dir)[0]
    finally:
        if stage.exists():
            shutil.rmtree(stage)


class ManagedLlamaServer:
    def __init__(
        self, profile, *, models_dir: Path | None = None, log_path: Path | None = None, timeout_s: float = 180
    ):
        self.profile = profile
        self.models_dir = models_dir
        self.log_path = log_path or Path("metrics/llama-server.log")
        self.timeout_s = timeout_s
        self.process = None
        self.log = None
        self.provenance = {}
        self.url = ""

    def start(self) -> str:
        executable, native = resolve_native(self.profile.backend, self.models_dir)
        model = resolve_model_path(self.profile.translation_model, models_dir=self.models_dir, local_only=True)
        if model is None:
            raise FileNotFoundError("Pinned E2B GGUF missing; run setup for the selected profile")
        entry = load_model_manifest()["models"][self.profile.translation_model]
        model_hash = sha256(Path(model))
        if model_hash != entry["sha256"]:
            raise ValueError("E2B GGUF checksum mismatch")
        # Choose an unused local port. A racing listener cannot be adopted:
        # /v1/models must return this unguessable per-child alias.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        alias = "stark-" + uuid.uuid4().hex
        self.url = f"http://127.0.0.1:{port}"
        command = [
            str(executable),
            "-m",
            model,
            "--alias",
            alias,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-ngl",
            "999" if self.profile.backend == "cuda" else "0",
            "-c",
            "512",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "-t",
            "3",
            "--parallel",
            "1",
        ]
        if self.profile.backend == "cpu":
            command += ["--no-kv-offload", "--no-op-offload"]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("ab")
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT
            )
        except BaseException:
            self.log.close()
            self.log = None
            raise
        atexit.register(self.stop)
        self.provenance = {
            "pid": self.process.pid,
            "url": self.url,
            "alias": alias,
            "command": command,
            "model": model,
            "model_sha256": model_hash,
            "native_revision": native["revision"],
            "native_executable_sha256": sha256(executable),
            "backend": self.profile.backend,
        }
        deadline = time.monotonic() + self.timeout_s
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Owned llama-server exited {self.process.returncode}; see {self.log_path}")
                try:
                    # self.url is assigned above from fixed 127.0.0.1 and a bound port.
                    with urllib.request.urlopen(self.url + "/health", timeout=1) as response:  # nosec B310
                        ready = json.load(response).get("status") == "ok"
                    if ready:
                        with urllib.request.urlopen(self.url + "/v1/models", timeout=1) as response:  # nosec B310
                            identities = {x.get("id") for x in json.load(response).get("data", [])}
                        if alias not in identities:
                            raise RuntimeError("llama-server identity mismatch; refusing foreign listener")
                        return self.url
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.1)
            raise TimeoutError(f"Owned llama-server did not become ready; see {self.log_path}")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        atexit.unregister(self.stop)
        if self.log is not None:
            self.log.close()
            self.log = None
