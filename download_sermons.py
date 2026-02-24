#!/usr/bin/env python3
"""
download_sermons.py -- yt-dlp wrapper for downloading church sermon audio from YouTube.

Downloads audio from Stark Road Gospel Hall's YouTube channel, converts to
16kHz mono WAV (Whisper's required input format), and stores results in
stark_data/raw/ with per-download metadata logging.

Supports:
  - YouTube channel URLs (all videos or recent N)
  - Playlist URLs
  - Individual video URLs (one or more)
  - Text files containing one URL per line

Usage examples:
  # Download from default Stark Road Gospel playlist (no args needed)
  python download_sermons.py

  # Dry run to preview what would be downloaded
  python download_sermons.py --dry-run

  # Download only 5 most recent videos for testing
  python download_sermons.py -n 5

  # Download with a different playlist
  python download_sermons.py --playlist-url "https://www.youtube.com/playlist?list=PLxxx"

  # Download specific videos
  python download_sermons.py -s "https://youtu.be/abc123" "https://youtu.be/def456"

  # Download from a URL list file, only videos after a date
  python download_sermons.py -s urls.txt --after-date 20240101

  # Download all videos from a channel
  python download_sermons.py -s "https://www.youtube.com/@StarkRoadGospelHall"
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLtTHU_srjk52WQQzZYzZiagLbKrS4BowV"
DEFAULT_OUTPUT_DIR = "stark_data/raw"
DEFAULT_METADATA_FILE = "stark_data/download_log.jsonl"
ACCENT_CHOICES = ["midwest", "scottish", "canadian", "westcoast", "british", "general"]
DEFAULT_MIN_DURATION_MIN = 10
DEFAULT_MAX_DURATION_MIN = 180
DEFAULT_RETRIES = 3
SLEEP_BETWEEN_DOWNLOADS = 2.0  # seconds -- basic rate limiting
SLEEP_BETWEEN_RETRIES = 5.0  # seconds

# 16kHz mono 16-bit PCM = 32,000 bytes per second of audio
WAV_BYTES_PER_SECOND = 16000 * 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check_dependencies():
    """Verify that yt-dlp and ffmpeg are available on PATH."""
    missing = []
    for tool in ("yt-dlp", "ffmpeg"):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        print(f"ERROR: Required tool(s) not found: {', '.join(missing)}", file=sys.stderr)
        print("Install them before running this script:", file=sys.stderr)
        for tool in missing:
            if tool == "yt-dlp":
                print("  pip install yt-dlp   OR   brew install yt-dlp", file=sys.stderr)
            elif tool == "ffmpeg":
                print("  brew install ffmpeg  OR   apt install ffmpeg", file=sys.stderr)
        sys.exit(1)


def sanitize_filename(name: str) -> str:
    """Remove or replace characters that are unsafe in filenames."""
    # Replace common problematic characters with underscores
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    # Collapse multiple underscores/spaces
    name = re.sub(r"[_\s]+", "_", name)
    # Strip leading/trailing underscores and dots
    name = name.strip("_. ")
    # Truncate to a reasonable length (leave room for video ID suffix + extension)
    if len(name) > 150:
        name = name[:150]
    return name


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string like '1h23m' or '45m12s'."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def format_size(size_bytes: int | float) -> str:
    """Format byte count into a human-readable string."""
    size_bytes = int(size_bytes)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def parse_sources(source_args: list[str]) -> list[str]:
    """
    Resolve source arguments into a flat list of URLs.

    Each argument can be:
      - A YouTube URL (returned as-is)
      - A path to a text file containing one URL per line
    """
    urls = []
    for src in source_args:
        # Check if it looks like a file path (exists on disk)
        if os.path.isfile(src):
            with open(src, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        urls.append(line)
        else:
            # Treat it as a URL
            urls.append(src)
    return urls


def get_video_list(url: str, args: argparse.Namespace) -> list[dict]:
    """
    Use yt-dlp --flat-playlist --dump-json to enumerate videos from a URL.

    Returns a list of dicts with at least: id, title, url, duration, upload_date.
    For single videos the list has one element.
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--ignore-errors",
    ]

    # Date filters
    if args.after_date:
        cmd += ["--dateafter", args.after_date]
    if args.before_date:
        cmd += ["--datebefore", args.before_date]

    # Limit number of videos
    if args.max_videos is not None:
        cmd += ["--playlist-end", str(args.max_videos)]

    cmd.append(url)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    videos = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        # flat-playlist entries have minimal info; normalise fields
        video_id = entry.get("id", "")
        title = entry.get("title") or entry.get("fulltitle") or video_id
        duration = entry.get("duration") or 0
        upload_date = entry.get("upload_date") or ""
        channel = entry.get("channel") or entry.get("uploader") or ""
        video_url = entry.get("url") or entry.get("webpage_url") or ""

        # If url field is just the video id (common in flat-playlist mode),
        # construct a full URL.
        if video_url and not video_url.startswith("http"):
            video_url = f"https://www.youtube.com/watch?v={video_id}"
        if not video_url:
            video_url = f"https://www.youtube.com/watch?v={video_id}"

        videos.append(
            {
                "id": video_id,
                "title": title,
                "duration": duration,
                "upload_date": upload_date,
                "channel": channel,
                "url": video_url,
            }
        )

    return videos


def filter_by_duration(videos: list[dict], min_sec: float, max_sec: float) -> tuple[list[dict], list[dict]]:
    """
    Split videos into (kept, skipped) based on duration bounds.

    Videos with duration == 0 (unknown) are kept -- their duration will be
    checked again after full metadata fetch during download.
    """
    kept, skipped = [], []
    for v in videos:
        dur = v.get("duration") or 0
        if dur == 0:
            # Unknown duration from flat-playlist; keep and check later
            kept.append(v)
        elif min_sec <= dur <= max_sec:
            kept.append(v)
        else:
            skipped.append(v)
    return kept, skipped


def fetch_video_description(video_url: str) -> str:
    """Fetch the description field from a single video via yt-dlp."""
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                "--no-warnings",
                "--ignore-errors",
                video_url,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip().splitlines()[0])
            return data.get("description", "")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        pass
    return ""


def save_video_metadata(
    output_dir: Path, video: dict, wav_path: Path, description: str = "", accent: str = "general"
) -> Path:
    """
    Save per-video metadata as a JSON file alongside the WAV file.

    Returns the path to the written JSON file.
    """
    json_path = wav_path.with_suffix(".json")

    # Extract speaker info from description
    speakers = ""
    if description:
        # Descriptions typically start with "Speakers: ..." or "Speaker: ..."
        m = re.match(r"Speakers?:\s*(.+?)(?:\.|$|\n)", description, re.IGNORECASE)
        if m:
            speakers = m.group(1).strip()

    metadata = {
        "video_id": video.get("id", ""),
        "title": video.get("title", ""),
        "upload_date": video.get("upload_date", ""),
        "duration_seconds": video.get("duration", 0),
        "channel": video.get("channel", ""),
        "url": video.get("url", ""),
        "speakers": speakers,
        "accent": accent,
        "description": description,
        "wav_filename": wav_path.name,
        "wav_path": str(wav_path),
        "download_timestamp": datetime.now(UTC).isoformat(),
        "audio_format": "16kHz mono PCM s16le WAV",
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16,
    }

    if wav_path.exists():
        stat = wav_path.stat()
        metadata["filesize_bytes"] = stat.st_size

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return json_path


def download_and_convert(
    video: dict,
    output_dir: Path,
    audio_format: str,
    skip_existing: bool,
    retries: int,
    save_per_video_json: bool = True,
    accent: str = "general",
) -> dict | None:
    """
    Download a single video's audio and convert to 16kHz mono WAV.

    Returns a metadata dict on success, or None on failure.
    """
    video_id = video["id"]
    title = video.get("title") or video_id
    safe_title = sanitize_filename(title)
    output_filename = f"{safe_title}_{video_id}.wav"
    output_path = output_dir / output_filename

    # Skip if already exists
    if skip_existing and output_path.exists():
        print(f"  SKIP (already exists): {output_path.name}")
        # Return metadata for the existing file
        stat = output_path.stat()
        return {
            "video_id": video_id,
            "title": title,
            "upload_date": video.get("upload_date", ""),
            "duration_seconds": video.get("duration", 0),
            "channel": video.get("channel", ""),
            "url": video["url"],
            "accent": accent,
            "output_path": str(output_path),
            "download_timestamp": datetime.now(UTC).isoformat(),
            "filesize_bytes": stat.st_size,
            "status": "skipped_existing",
        }

    # Temporary download path (yt-dlp will add its own extension)
    temp_template = str(output_dir / f".tmp_{video_id}.%(ext)s")
    wav_path_temp = output_dir / f".tmp_{video_id}.wav"

    # Build yt-dlp command -- download best quality audio only
    ytdlp_cmd = [
        "yt-dlp",
        "--no-warnings",
        "--ignore-errors",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",  # best quality
        "-f",
        audio_format,
        "--output",
        temp_template,
        "--no-playlist",  # ensure we download just this video
        "--retries",
        str(retries),
        "--fragment-retries",
        str(retries),
        "--concurrent-fragments",
        "4",  # speed up downloads
    ]
    ytdlp_cmd.append(video["url"])

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ytdlp_cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                last_error = result.stderr.strip() or result.stdout.strip()
                # Check for unrecoverable errors
                if any(
                    phrase in last_error.lower()
                    for phrase in [
                        "private video",
                        "video unavailable",
                        "this video is not available",
                        "sign in to confirm your age",
                        "removed by the uploader",
                    ]
                ):
                    print(f"  UNAVAILABLE: {last_error[:120]}")
                    return None
                if attempt < retries:
                    print(f"  Retry {attempt}/{retries}: {last_error[:100]}")
                    time.sleep(SLEEP_BETWEEN_RETRIES)
                    continue
                print(f"  FAILED after {retries} attempts: {last_error[:120]}")
                return None

            # yt-dlp succeeded -- find the downloaded WAV
            if not wav_path_temp.exists():
                # Sometimes the extension differs; look for any temp file
                candidates = list(output_dir.glob(f".tmp_{video_id}.*"))
                if not candidates:
                    print("  ERROR: yt-dlp completed but no output file found")
                    return None
                # Use the first candidate and convert
                source_file = candidates[0]
            else:
                source_file = wav_path_temp

            # Convert to 16kHz mono WAV using ffmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # overwrite output
                "-i",
                str(source_file),  # input
                "-ar",
                "16000",  # 16kHz sample rate
                "-ac",
                "1",  # mono
                "-c:a",
                "pcm_s16le",  # 16-bit signed little-endian PCM
                str(output_path),
            ]

            ffmpeg_result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if ffmpeg_result.returncode != 0:
                print(f"  ERROR: ffmpeg conversion failed: {ffmpeg_result.stderr[:120]}")
                # Clean up partial output
                output_path.unlink(missing_ok=True)
                return None

            # Clean up temp files
            _cleanup_temp_files(output_dir, video_id)

            # Gather metadata
            stat = output_path.stat()

            # Get accurate duration from the converted WAV via ffprobe
            duration = _get_wav_duration(output_path) or video.get("duration", 0)

            # Fetch description for speaker info and save per-video JSON
            description = ""
            if save_per_video_json:
                description = fetch_video_description(video["url"])
                save_video_metadata(output_dir, video, output_path, description, accent=accent)

            return {
                "video_id": video_id,
                "title": title,
                "upload_date": video.get("upload_date", ""),
                "duration_seconds": duration,
                "channel": video.get("channel", ""),
                "url": video["url"],
                "accent": accent,
                "description": description,
                "output_path": str(output_path),
                "download_timestamp": datetime.now(UTC).isoformat(),
                "filesize_bytes": stat.st_size,
                "status": "downloaded",
            }

        except subprocess.TimeoutExpired:
            last_error = "Command timed out"
            if attempt < retries:
                print(f"  Timeout, retrying ({attempt}/{retries})...")
                time.sleep(SLEEP_BETWEEN_RETRIES)
            else:
                print(f"  FAILED: Command timed out after {retries} attempts")
                _cleanup_temp_files(output_dir, video_id)
                return None

        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                print(f"  Error, retrying ({attempt}/{retries}): {exc}")
                time.sleep(SLEEP_BETWEEN_RETRIES)
            else:
                print(f"  FAILED: {exc}")
                _cleanup_temp_files(output_dir, video_id)
                return None

    return None


def _cleanup_temp_files(output_dir: Path, video_id: str):
    """Remove any temporary files left by a download attempt."""
    for tmp in output_dir.glob(f".tmp_{video_id}*"):
        try:
            tmp.unlink()
        except OSError:
            pass


def _get_wav_duration(wav_path: Path) -> float | None:
    """Get the duration of a WAV file in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return None


def append_metadata(metadata_file: Path, entry: dict):
    """Append a single JSON line to the metadata log file."""
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_existing_ids(metadata_file: Path) -> set[str]:
    """Load video IDs already present in the metadata log."""
    ids = set()
    if metadata_file.exists():
        with open(metadata_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    vid = entry.get("video_id")
                    if vid:
                        ids.add(vid)
                except json.JSONDecodeError:
                    continue
    return ids


def print_download_estimates(videos: list[dict]):
    """Print estimated download sizes and times for the given video list."""
    total_duration = sum(v.get("duration", 0) or 0 for v in videos)
    known_count = sum(1 for v in videos if (v.get("duration") or 0) > 0)
    unknown_count = len(videos) - known_count

    # WAV output size: 16kHz * 2 bytes * mono = 32KB/s
    wav_size = total_duration * WAV_BYTES_PER_SECOND

    # Compressed download from YouTube (typically ~128kbps for bestaudio m4a)
    compressed_size = total_duration * 128 * 1000 / 8  # bits to bytes

    # Estimate download time at various speeds
    speeds_mbps = [10, 50, 100]

    print("\n--- DOWNLOAD ESTIMATES ---")
    print(f"  Videos with known duration: {known_count}")
    if unknown_count > 0:
        print(f"  Videos with unknown duration: {unknown_count} (estimates may be low)")
    print(f"  Total audio duration: {format_duration(total_duration)} ({total_duration / 3600:.1f} hours)")
    print(f"  Estimated compressed download: ~{format_size(compressed_size)}")
    print(f"  Final WAV output size (16kHz mono): ~{format_size(wav_size)}")
    print("  Estimated download time:")
    for speed in speeds_mbps:
        dl_seconds = compressed_size / (speed * 1_000_000 / 8)
        # Add overhead for ffmpeg conversion (~0.5x real-time) + rate limiting
        convert_seconds = total_duration * 0.1  # ffmpeg is fast for resampling
        rate_limit_seconds = len(videos) * SLEEP_BETWEEN_DOWNLOADS
        total_time = dl_seconds + convert_seconds + rate_limit_seconds
        print(
            f"    At {speed:3d} Mbps: ~{format_duration(total_time)} "
            f"(download: {format_duration(dl_seconds)}, "
            f"convert: {format_duration(convert_seconds)}, "
            f"rate-limit: {format_duration(rate_limit_seconds)})"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download sermon audio from YouTube and convert to 16kHz mono WAV for Whisper.\n\n"
            "By default, downloads from the Stark Road Gospel Hall 'Gospel' playlist.\n"
            "Use --source/-s or --playlist-url to override."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source",
        "-s",
        nargs="+",
        default=None,
        help=(
            "YouTube URL(s) -- channel, playlist, or individual video URLs. "
            "Can also be a path to a text file with one URL per line. "
            "If omitted, uses the default Stark Road Gospel playlist."
        ),
    )
    source_group.add_argument(
        "--playlist-url",
        type=str,
        default=None,
        help=(f"YouTube playlist URL to download from. Default: {DEFAULT_PLAYLIST_URL}"),
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for WAV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-videos",
        "-n",
        type=int,
        default=None,
        help="Maximum number of videos to download (default: unlimited)",
    )
    parser.add_argument(
        "--after-date",
        type=str,
        default=None,
        help="Only videos published after this date (YYYYMMDD)",
    )
    parser.add_argument(
        "--before-date",
        type=str,
        default=None,
        help="Only videos published before this date (YYYYMMDD)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION_MIN,
        help=f"Minimum video duration in minutes (default: {DEFAULT_MIN_DURATION_MIN})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION_MIN,
        help=f"Maximum video duration in minutes (default: {DEFAULT_MAX_DURATION_MIN})",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="bestaudio",
        help="yt-dlp audio format selector (default: bestaudio)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip videos whose output WAV already exists (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--no-video-json",
        action="store_true",
        default=False,
        help="Skip saving per-video JSON metadata alongside WAV files",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=DEFAULT_METADATA_FILE,
        help=f"Path to write download metadata JSONL (default: {DEFAULT_METADATA_FILE})",
    )
    parser.add_argument(
        "--accent",
        type=str,
        choices=ACCENT_CHOICES,
        default="general",
        help=(
            "Accent label to tag all downloads from this run "
            f"(choices: {', '.join(ACCENT_CHOICES)}, default: general). "
            "Downloads are organized into stark_data/raw/{accent}/ subdirectories."
        ),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Dependency check
    check_dependencies()

    # Resolve paths relative to project root (script location)
    project_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    # Organize downloads into accent subdirectory
    output_dir = output_dir / args.accent
    metadata_file = Path(args.metadata_file)
    if not metadata_file.is_absolute():
        metadata_file = project_root / metadata_file

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine source URLs: --source, --playlist-url, or default playlist
    if args.source:
        urls = parse_sources(args.source)
    elif args.playlist_url:
        urls = [args.playlist_url]
    else:
        urls = [DEFAULT_PLAYLIST_URL]
        print(f"Using default playlist: {DEFAULT_PLAYLIST_URL}")

    if not urls:
        print("ERROR: No URLs provided.", file=sys.stderr)
        sys.exit(1)

    print(f"Sources: {len(urls)} URL(s)")
    print(f"Accent:  {args.accent}")
    print(f"Output:  {output_dir}")
    print(f"Duration filter: {args.min_duration}m - {args.max_duration}m")
    if args.after_date:
        print(f"After date: {args.after_date}")
    if args.before_date:
        print(f"Before date: {args.before_date}")
    if args.max_videos:
        print(f"Max videos: {args.max_videos}")
    print()

    # Enumerate all videos across all source URLs
    print("Enumerating videos...")
    all_videos = []
    seen_ids = set()
    for url in urls:
        print(f"  Scanning: {url}")
        try:
            videos = get_video_list(url, args)
            for v in videos:
                vid = v["id"]
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    all_videos.append(v)
        except subprocess.TimeoutExpired:
            print(f"  WARNING: Timed out scanning {url}, skipping.")
        except Exception as exc:
            print(f"  WARNING: Error scanning {url}: {exc}")

    if not all_videos:
        print("\nNo videos found. Check the URL(s) and try again.")
        sys.exit(0)

    print(f"\nFound {len(all_videos)} unique video(s)")

    # Duration filter
    min_sec = args.min_duration * 60
    max_sec = args.max_duration * 60
    videos, skipped_duration = filter_by_duration(all_videos, min_sec, max_sec)

    if skipped_duration:
        print(f"Filtered out {len(skipped_duration)} video(s) by duration:")
        for v in skipped_duration[:5]:
            dur = v.get("duration", 0)
            print(f"  - {v['title'][:60]} ({format_duration(dur)})")
        if len(skipped_duration) > 5:
            print(f"  ... and {len(skipped_duration) - 5} more")

    # Apply max-videos limit (after filtering)
    if args.max_videos is not None and len(videos) > args.max_videos:
        videos = videos[: args.max_videos]

    print(f"\n{len(videos)} video(s) to process")

    # Show download estimates
    print_download_estimates(videos)

    # Dry run -- print and exit
    if args.dry_run:
        print("\n--- DRY RUN ---")
        total_duration = 0
        for i, v in enumerate(videos, 1):
            dur = v.get("duration", 0)
            total_duration += dur
            dur_str = format_duration(dur) if dur else "??m"
            date_str = v.get("upload_date", "????????")
            print(f"  {i:3d}. [{date_str}] {v['title'][:70]} ({dur_str}) [{v['id']}]")
        print(f"\nTotal estimated duration: {format_duration(total_duration)}")
        print("No files were downloaded (dry run).")
        return

    # Download loop
    print()
    results = {
        "downloaded": [],
        "skipped": [],
        "failed": [],
    }
    total_count = len(videos)
    start_time = time.monotonic()

    for i, video in enumerate(videos, 1):
        title = video.get("title", video["id"])
        dur = video.get("duration", 0)
        dur_str = format_duration(dur) if dur else "??m"

        # Progress with ETA
        elapsed = time.monotonic() - start_time
        if i > 1 and elapsed > 0:
            rate = (i - 1) / elapsed
            remaining = (total_count - i + 1) / rate
            eta_str = f" [ETA: {format_duration(remaining)}]"
        else:
            eta_str = ""

        print(f"[{i}/{total_count}]{eta_str} {title[:65]} ({dur_str})")

        meta = download_and_convert(
            video=video,
            output_dir=output_dir,
            audio_format=args.format,
            skip_existing=args.skip_existing,
            retries=DEFAULT_RETRIES,
            save_per_video_json=not args.no_video_json,
            accent=args.accent,
        )

        if meta is None:
            results["failed"].append(video)
            print(f"  FAILED: {video['id']}")
        elif meta.get("status") == "skipped_existing":
            results["skipped"].append(meta)
            append_metadata(metadata_file, meta)
        else:
            results["downloaded"].append(meta)
            append_metadata(metadata_file, meta)
            size_str = format_size(meta.get("filesize_bytes", 0))
            print(f"  OK: {Path(meta['output_path']).name} ({size_str})")

        # Rate limiting between downloads
        if i < total_count:
            time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    # Summary
    total_elapsed = time.monotonic() - start_time
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    downloaded = results["downloaded"]
    skipped = results["skipped"]
    failed = results["failed"]

    total_downloaded_duration = sum(m.get("duration_seconds", 0) for m in downloaded)
    total_downloaded_size = sum(m.get("filesize_bytes", 0) for m in downloaded)

    print(f"  Downloaded:  {len(downloaded)}")
    print(f"  Skipped:     {len(skipped)} (already existed)")
    print(f"  Failed:      {len(failed)}")
    print(f"  Total duration downloaded: {format_duration(total_downloaded_duration)}")
    print(f"  Total size downloaded:     {format_size(total_downloaded_size)}")
    print(f"  Wall clock time:           {format_duration(total_elapsed)}")
    print(f"  Output directory:          {output_dir}")
    print(f"  Metadata log:              {metadata_file}")

    if failed:
        print("\nFailed videos:")
        for v in failed:
            print(f"  - {v.get('title', v['id'])[:70]} [{v['id']}]")

    print()


if __name__ == "__main__":
    main()
