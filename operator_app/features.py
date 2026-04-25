"""Operator-side wrappers for the post-processing features (Phase 9.6).

The three feature modules in ``features/`` (verse extraction, sermon
summary, diarization) were originally batch CLI tools. This module exposes
two of them to the live operator UI:

- **Live verse highlights** — runs ``VerseExtractor`` over the rolling tail
  of the session CSV every few seconds and surfaces the latest references
  to the operator. Cheap (regex + book-name resolution, no LLM), so it's
  fine to do inline.
- **Post-session summary trigger** — kicks off ``features/summarize_sermon.py``
  as a subprocess against the just-finished session CSV. The summarization
  model loads a local LLM (Gemma 4B-class) which we don't want competing
  with the live pipeline for VRAM, so it always runs out-of-process. The
  operator UI polls a task ID for status / result.

Live diarization is intentionally out of scope: it requires audio files
(not just transcripts) and a 2–4 GB pyannote model load — too heavy for
the live path. Documented as 9.6.1 in the plan; the existing CLI continues
to handle it post-service.
"""

from __future__ import annotations

import csv
import logging
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VerseHighlight:
    chunk_id: int
    reference: str
    context: str
    timestamp: float | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class SummaryTask:
    task_id: str
    csv_path: str
    output_path: str
    state: str = "pending"  # pending | running | done | error
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    return_code: int | None = None
    error: str | None = None
    result: dict | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# -- live verse highlights ---------------------------------------------------


class VerseHighlightWatcher:
    """Periodically tails a session CSV and runs VerseExtractor over new rows.

    Stateful: keeps a single ``VerseExtractor`` instance for the life of the
    session so context tracking ("verse 5" after "Romans 8") works across
    successive polls.
    """

    POLL_INTERVAL_S = 4.0
    MAX_HIGHLIGHTS = 50  # ring buffer of recent finds

    def __init__(self, csv_path: Path | str, project_root: Path | None = None) -> None:
        self._csv_path = Path(csv_path)
        self._project_root = project_root or Path.cwd()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._highlights: list[VerseHighlight] = []
        self._last_chunk_id = -1
        self._extractor = None  # lazy import — features/extract_verses pulls in regex tables

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="verse-watcher", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def snapshot(self, since_chunk: int | None = None) -> list[dict]:
        with self._lock:
            highlights = list(self._highlights)
        if since_chunk is not None:
            highlights = [h for h in highlights if h.chunk_id > since_chunk]
        return [h.to_dict() for h in highlights]

    def force_scan(self) -> list[dict]:
        """One-shot scan; used by the ``GET /api/features/verses`` endpoint."""
        self._scan_once()
        return self.snapshot()

    def _ensure_extractor(self) -> None:
        if self._extractor is not None:
            return
        try:
            from features.extract_verses import VerseExtractor

            self._extractor = VerseExtractor()
        except Exception as exc:
            logger.warning("VerseExtractor import failed: %s", exc)

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=self.POLL_INTERVAL_S):
            self._scan_once()

    def _scan_once(self) -> None:
        self._ensure_extractor()
        if self._extractor is None or not self._csv_path.exists():
            return
        try:
            with self._csv_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except OSError:
            return

        new_finds: list[VerseHighlight] = []
        for row in rows:
            try:
                chunk_id = int(float(row.get("chunk_id", 0) or 0))
            except (TypeError, ValueError):
                continue
            if chunk_id <= self._last_chunk_id:
                continue
            text = str(row.get("english", "") or "")
            if not text:
                continue
            # Process this row through the stateful extractor; collect its
            # NEW references (those added during this call).
            before = len(self._extractor.references)
            try:
                self._extractor.extract_from_text(
                    text,
                    timestamp=str(row.get("timestamp", "") or ""),
                    speaker=row.get("speaker"),
                )
            except Exception as exc:
                logger.debug("verse extract failed for chunk %s: %s", chunk_id, exc)
                continue
            for ref in self._extractor.references[before:]:
                new_finds.append(
                    VerseHighlight(
                        chunk_id=chunk_id,
                        reference=ref.get("reference", "?"),
                        context=ref.get("context", "") or text[:160],
                        timestamp=_safe_float(row.get("timestamp")),
                    )
                )
            self._last_chunk_id = chunk_id

        if new_finds:
            with self._lock:
                self._highlights.extend(new_finds)
                # Cap ring buffer
                if len(self._highlights) > self.MAX_HIGHLIGHTS:
                    self._highlights = self._highlights[-self.MAX_HIGHLIGHTS :]


def _safe_float(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


# -- post-session summary trigger --------------------------------------------


class SummaryTaskRunner:
    """Spawns ``features/summarize_sermon.py`` subprocesses; tracks tasks by id."""

    SUBPROCESS_TIMEOUT_S = 600  # 10 minutes for LLM-based summary

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = project_root or Path.cwd()
        self._lock = threading.RLock()
        self._tasks: dict[str, SummaryTask] = {}

    def submit(self, csv_path: str, output_path: str | None = None) -> SummaryTask:
        task_id = uuid.uuid4().hex[:12]
        if output_path is None:
            csv_p = Path(csv_path)
            output_path = str(csv_p.parent / f"summary_{csv_p.stem.replace('ab_metrics_', '')}.json")
        task = SummaryTask(task_id=task_id, csv_path=csv_path, output_path=output_path)
        with self._lock:
            self._tasks[task_id] = task
        thread = threading.Thread(target=self._run, args=(task,), name=f"summary-{task_id}", daemon=True)
        thread.start()
        return task

    def get(self, task_id: str) -> SummaryTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[SummaryTask]:
        with self._lock:
            return list(self._tasks.values())

    def _run(self, task: SummaryTask) -> None:
        try:
            with self._lock:
                task.state = "running"

            argv = [
                sys.executable,
                "-u",
                str(self._project_root / "features" / "summarize_sermon.py"),
                "--input",
                task.csv_path,
                "--output",
                task.output_path,
            ]
            logger.info("summary task %s: spawning %s", task.task_id, " ".join(argv))
            try:
                completed = subprocess.run(
                    argv,
                    cwd=str(self._project_root),
                    capture_output=True,
                    text=True,
                    timeout=self.SUBPROCESS_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                with self._lock:
                    task.state = "error"
                    task.error = f"subprocess exceeded {self.SUBPROCESS_TIMEOUT_S}s timeout"
                    task.finished_at = time.time()
                return

            with self._lock:
                task.return_code = completed.returncode
                task.finished_at = time.time()
                if completed.returncode != 0:
                    task.state = "error"
                    task.error = (completed.stderr or "")[-500:]
                    return
                task.state = "done"
                # Try to load the JSON output; if it parses, attach.
                try:
                    import json

                    if Path(task.output_path).exists():
                        task.result = json.loads(Path(task.output_path).read_text())
                except Exception as exc:
                    task.error = f"output parse failed: {exc}"
        except Exception as exc:
            logger.exception("summary task %s crashed", task.task_id)
            with self._lock:
                task.state = "error"
                task.error = f"{type(exc).__name__}: {exc}"
                task.finished_at = time.time()


# -- module-level singletons -------------------------------------------------


_verse_watcher: VerseHighlightWatcher | None = None
_summary_runner: SummaryTaskRunner | None = None
_lock = threading.Lock()


def get_verse_watcher(
    csv_path: Path | str | None = None, project_root: Path | None = None
) -> VerseHighlightWatcher | None:
    """Lazily create / rebind the watcher to a session's CSV.

    Pass csv_path=None to retrieve the current watcher (or None if no
    session has been started). Pass csv_path=<new path> to rebind it; this
    is what the operator pipeline runner does when starting a session.
    """
    global _verse_watcher
    with _lock:
        if csv_path is None:
            return _verse_watcher
        # rebind: stop existing, start fresh
        if _verse_watcher is not None:
            try:
                _verse_watcher.stop()
            except Exception:
                pass
        _verse_watcher = VerseHighlightWatcher(csv_path=csv_path, project_root=project_root)
        _verse_watcher.start()
        return _verse_watcher


def get_summary_runner(project_root: Path | None = None) -> SummaryTaskRunner:
    global _summary_runner
    with _lock:
        if _summary_runner is None:
            _summary_runner = SummaryTaskRunner(project_root=project_root)
        return _summary_runner


def reset_features_for_tests() -> None:
    global _verse_watcher, _summary_runner
    with _lock:
        if _verse_watcher is not None:
            try:
                _verse_watcher.stop()
            except Exception:
                pass
        _verse_watcher = None
        _summary_runner = None
