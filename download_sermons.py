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
  # Download all videos from a channel
  python download_sermons.py -s "https://www.youtube.com/@StarkRoadGospelHall"

  # Download a specific playlist, max 20 videos
  python download_sermons.py -s "https://www.youtube.com/playlist?list=PLxxx" -n 20

  # Download specific videos
  python download_sermons.py -s "https://youtu.be/abc123" "https://youtu.be/def456"

  # Download from a URL list file, only videos after a date
  python download_sermons.py -s urls.txt --after-date 20240101

  # Dry run to see what would be downloaded
  python download_sermons.py -s "https://www.youtube.com/@StarkRoadGospelHall" --dry-run
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "stark_data/raw"
DEFAULT_METADATA_FILE = "stark_data/download_log.jsonl"
DEFAULT_MIN_DURATION_MIN = 10
DEFAULT_MAX_DURATION_MIN = 180
DEFAULT_RETRIES = 3
SLEEP_BETWEEN_DOWNLOADS = 2.0  # seconds -- basic rate limiting
SLEEP_BETWEEN_RETRIES = 5.0   # seconds


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
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Collapse multiple underscores/spaces
    name = re.sub(r'[_\s]+', '_', name)
    # Strip leading/trailing underscores and dots
    name = name.strip('_. ')
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


def format_size(size_bytes: int) -> str:
    """Format byte count into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


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
            with open(src, "r", encoding="utf-8") as f:
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

        videos.append({
            "id": video_id,
            "title": title,
            "duration": duration,
            "upload_date": upload_date,
            "channel": channel,
            "url": video_url,
        })

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


def download_and_convert(
    video: dict,
    output_dir: Path,
    audio_format: str,
    skip_existing: bool,
    retries: int,
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
            "output_path": str(output_path),
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "filesize_bytes": stat.st_size,
            "status": "skipped_existing",
        }

    # Temporary download path (yt-dlp will add its own extension)
    temp_template = str(output_dir / f".tmp_{video_id}.%(ext)s")
    wav_path_temp = output_dir / f".tmp_{video_id}.wav"

    # Build yt-dlp command -- download audio only
    ytdlp_cmd = [
        "yt-dlp",
        "--no-warnings",
        "--ignore-errors",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-f", audio_format,
        "--output", temp_template,
        "--no-playlist",  # ensure we download just this video
        "--retries", str(retries),
        "--fragment-retries", str(retries),
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
                if any(phrase in last_error.lower() for phrase in [
                    "private video", "video unavailable", "this video is not available",
                    "sign in to confirm your age", "removed by the uploader",
                ]):
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
                    print(f"  ERROR: yt-dlp completed but no output file found")
                    return None
                # Use the first candidate and convert
                source_file = candidates[0]
            else:
                source_file = wav_path_temp

            # Convert to 16kHz mono WAV using ffmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",                    # overwrite output
                "-i", str(source_file),  # input
                "-ar", "16000",          # 16kHz sample rate
                "-ac", "1",              # mono
                "-c:a", "pcm_s16le",     # 16-bit signed little-endian PCM
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

            return {
                "video_id": video_id,
                "title": title,
                "upload_date": video.get("upload_date", ""),
                "duration_seconds": duration,
                "channel": video.get("channel", ""),
                "url": video["url"],
                "output_path": str(output_path),
                "download_timestamp": datetime.now(timezone.utc).isoformat(),
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
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
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
        with open(metadata_file, "r", encoding="utf-8") as f:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download sermon audio from YouTube and convert to 16kHz mono WAV for Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--source", "-s",
        nargs="+",
        required=True,
        help=(
            "YouTube URL(s) -- channel, playlist, or individual video URLs. "
            "Can also be a path to a text file with one URL per line."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for WAV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-videos", "-n",
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
        "--metadata-file",
        type=str,
        default=DEFAULT_METADATA_FILE,
        help=f"Path to write download metadata JSONL (default: {DEFAULT_METADATA_FILE})",
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
    metadata_file = Path(args.metadata_file)
    if not metadata_file.is_absolute():
        metadata_file = project_root / metadata_file

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse all source URLs
    urls = parse_sources(args.source)
    if not urls:
        print("ERROR: No URLs provided.", file=sys.stderr)
        sys.exit(1)

    print(f"Sources: {len(urls)} URL(s)")
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
        videos = videos[:args.max_videos]

    print(f"\n{len(videos)} video(s) to process")

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

    for i, video in enumerate(videos, 1):
        title = video.get("title", video["id"])
        dur = video.get("duration", 0)
        dur_str = format_duration(dur) if dur else "??m"
        print(f"Downloading {i}/{total_count}: {title[:70]} ({dur_str})")

        meta = download_and_convert(
            video=video,
            output_dir=output_dir,
            audio_format=args.format,
            skip_existing=args.skip_existing,
            retries=DEFAULT_RETRIES,
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
    print(f"  Output directory:          {output_dir}")
    print(f"  Metadata log:              {metadata_file}")

    if failed:
        print(f"\nFailed videos:")
        for v in failed:
            print(f"  - {v.get('title', v['id'])[:70]} [{v['id']}]")

    print()


if __name__ == "__main__":
    main()
