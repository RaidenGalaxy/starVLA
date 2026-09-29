"""Resumably download selected public media under a strict local byte quota."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .protocol import atomic_json, atomic_jsonl, file_sha256


class DownloadTimeoutError(TimeoutError):
    pass


def _download_timeout(_signum: int, _frame: object) -> None:
    raise DownloadTimeoutError("media request exceeded wall-time limit")


def _load_jobs(paths: list[Path]) -> list[dict[str, Any]]:
    jobs = []
    seen: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                output = str(Path(row["output_path"]).expanduser().resolve())
                if output in seen:
                    continue
                seen.add(output)
                row["output_path"] = output
                row["job_manifest"] = str(path.resolve())
                jobs.append(row)
    return jobs


def _probe_video(path: Path) -> dict[str, float | int]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("OpenCV is required to validate downloaded videos") from error
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open downloaded video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = capture.read()
    finally:
        capture.release()
    if not ok or fps <= 0 or frames <= 0:
        raise ValueError(f"downloaded video has invalid metadata: {path}")
    return {"fps": fps, "frames": frames, "duration_sec": frames / fps, "bytes": path.stat().st_size}


def _hf_root(job: dict[str, Any]) -> Path:
    output = Path(job["output_path"])
    parts = Path(job["filename"]).parts
    if tuple(output.parts[-len(parts) :]) != parts:
        raise ValueError(f"HF output path does not end in repository filename: {output}")
    return output.parents[len(parts) - 1]


def _download_huggingface(job: dict[str, Any]) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    result = Path(
        hf_hub_download(
            repo_id=str(job["repo_id"]),
            repo_type=str(job.get("repo_type", "dataset")),
            filename=str(job["filename"]),
            local_dir=str(_hf_root(job)),
        )
    ).resolve()
    expected = Path(job["output_path"]).resolve()
    if result != expected or not expected.exists():
        raise RuntimeError(f"unexpected Hugging Face target: {result} != {expected}")


def _download_youtube(job: dict[str, Any]) -> None:
    try:
        import imageio_ffmpeg
        import yt_dlp
        from yt_dlp.utils import download_range_func
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install yt-dlp and imageio-ffmpeg for FPS-Bench media") from error

    target = Path(job["output_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fpsbench-download-", dir=target.parent) as temporary_name:
        temporary = Path(temporary_name)
        ffmpeg_dir = temporary / "ffmpeg-bin"
        ffmpeg_dir.mkdir()
        # imageio ships a versioned executable name, while yt-dlp discovers
        # only a binary literally named ``ffmpeg`` inside --ffmpeg-location.
        os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg_dir / "ffmpeg")
        options = {
            "format": (
                "best[height<=720][ext=mp4][protocol^=http]/"
                "bestvideo[height<=720][ext=mp4][protocol^=http]/"
                "best[height<=720][protocol^=http]/"
                "bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]/"
                "bestvideo[height<=720]/best[height<=720]/best"
            ),
            "outtmpl": str(temporary / "clip.%(ext)s"),
            "download_ranges": download_range_func(
                None,
                [(float(job["start_sec"]), float(job["end_sec"]))],
            ),
            "force_keyframes_at_cuts": True,
            "ffmpeg_location": str(ffmpeg_dir),
            "noplaylist": True,
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
            "quiet": True,
            "no_warnings": True,
        }
        previous_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{previous_path}"
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                code = downloader.download([str(job["url"])])
        finally:
            os.environ["PATH"] = previous_path
        if code:
            raise RuntimeError(f"yt-dlp exited with status {code}")
        candidates = [
            item
            for item in temporary.iterdir()
            if item.is_file() and not item.name.endswith((".part", ".ytdl"))
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one downloaded clip, found {candidates}")
        source = candidates[0]
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp.mp4")
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(staged),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode:
            if staged.exists():
                staged.unlink()
            raise RuntimeError(f"ffmpeg failed: {completed.stderr[-1000:]}")
        os.replace(staged, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-gb", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backend", choices=["all", "huggingface", "yt_dlp"], default="all")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--download-timeout-sec", type=int, default=600)
    args = parser.parse_args()
    if args.budget_gb <= 0 or args.retries < 0 or args.download_timeout_sec <= 0:
        parser.error("budget must be positive and retries non-negative")
    jobs = _load_jobs(args.jobs)
    if args.backend != "all":
        jobs = [job for job in jobs if job["backend"] == args.backend]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    if not jobs:
        raise ValueError("no download jobs selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quota = int(args.budget_gb * 1024**3)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    used = sum(
        Path(job["output_path"]).stat().st_size
        for job in jobs
        if Path(job["output_path"]).is_file()
    )
    started = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, _download_timeout)
    try:
        for index, job in enumerate(jobs):
            target = Path(job["output_path"])
            if used >= quota and not target.exists():
                failures.append({"sample_id": job["sample_id"], "error": "media byte quota reached", "retryable": True})
                break
            record: dict[str, Any] | None = None
            if target.exists():
                try:
                    probe = _probe_video(target)
                    record = {**job, **probe, "status": "ok", "cached": True, "sha256": file_sha256(target)}
                except ValueError as error:
                    failures.append(
                        {"sample_id": job["sample_id"], "output_path": str(target), "error": str(error), "retryable": False}
                    )
            else:
                last_error: Exception | None = None
                for attempt in range(args.retries + 1):
                    try:
                        signal.setitimer(signal.ITIMER_REAL, args.download_timeout_sec)
                        if job["backend"] == "huggingface":
                            _download_huggingface(job)
                        elif job["backend"] == "yt_dlp":
                            _download_youtube(job)
                        else:
                            raise ValueError(f"unknown backend: {job['backend']}")
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        probe = _probe_video(target)
                        used += int(probe["bytes"])
                        record = {
                            **job,
                            **probe,
                            "status": "ok",
                            "cached": False,
                            "attempt": attempt + 1,
                            "sha256": file_sha256(target),
                        }
                        break
                    except Exception as error:  # network/backend failures are recorded and resumable
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        last_error = error
                        if attempt < args.retries:
                            time.sleep(min(2 ** attempt, 4))
                if record is None:
                    failures.append(
                        {
                            "sample_id": job["sample_id"],
                            "backend": job["backend"],
                            "output_path": str(target),
                            "error": f"{type(last_error).__name__}: {last_error}",
                            "retryable": True,
                        }
                    )
            if record is not None:
                records.append(record)
            atomic_jsonl(args.output_dir / "downloads.jsonl", records)
            atomic_jsonl(args.output_dir / "failures.jsonl", failures)
            elapsed = time.monotonic() - started
            rate = (index + 1) / elapsed if elapsed else 0.0
            eta = (len(jobs) - index - 1) / rate if rate else 0.0
            print(
                f"[{index + 1}/{len(jobs)}] {job['sample_id']} status={'ok' if record else 'failed'} "
                f"used={used / 1024**3:.3f}GiB eta={eta:.1f}s",
                flush=True,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    atomic_json(
        args.output_dir / "download_run.json",
        {
            "format_version": 1,
            "job_manifests": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in args.jobs
            ],
            "backend": args.backend,
            "jobs_requested": len(jobs),
            "completed": len(records),
            "failures": len(failures),
            "bytes": used,
            "budget_bytes": quota,
            "runtime_seconds": time.monotonic() - started,
            "download_timeout_sec": args.download_timeout_sec,
        },
    )
    print(f"completed {len(records)}/{len(jobs)} downloads; {len(failures)} failures")


if __name__ == "__main__":
    main()
