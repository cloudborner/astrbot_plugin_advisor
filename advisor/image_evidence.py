from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import os
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image as PILImage

from .analysis_draft import DraftImage

MAX_LOCAL_IMAGE_BYTES = 32 * 1024 * 1024
MAX_REMOTE_IMAGE_BYTES = 12 * 1024 * 1024
MAX_REMOTE_TOTAL_BYTES = 32 * 1024 * 1024
MAX_IMAGE_LONG_EDGE = 2048
MAX_DECODED_PIXELS = 40_000_000
RESIZE_FILE_THRESHOLD_BYTES = 4 * 1024 * 1024
_VOLATILE_QUERY_KEYS = {
    "auth_key",
    "expire",
    "expires",
    "rkey",
    "sign",
    "signature",
    "token",
}


@dataclass(frozen=True, slots=True)
class PreparedImage:
    evidence_id: str
    reference: str
    message_evidence_id: str
    timestamp: int | None
    fingerprint: str
    context_weight: int = 0


@dataclass(frozen=True, slots=True)
class ImagePreparationResult:
    images: tuple[PreparedImage, ...]
    invalid_count: int
    duplicate_count: int
    checked_remote_count: int = 0
    downloaded_bytes: int = 0
    temporary_paths: tuple[str, ...] = ()

    def cleanup(self) -> None:
        for value in self.temporary_paths:
            try:
                Path(value).unlink(missing_ok=True)
            except Exception:
                pass

    def __del__(self) -> None:
        self.cleanup()


@dataclass(frozen=True, slots=True)
class RemoteProbeResult:
    fingerprint: str
    byte_count: int
    reference: str
    temporary_path: str = ""


def _is_image_header(header: bytes) -> bool:
    return (
        header.startswith(
            (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"BM")
        )
        or (header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP")
    )


def _remote_fingerprint(reference: str) -> str:
    parsed = urlsplit(reference)
    stable_query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.casefold() not in _VOLATILE_QUERY_KEYS
        )
    )
    stable = urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            parsed.path,
            stable_query,
            "",
        )
    )
    return "remote:" + hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _local_image_hash(path: Path) -> str | None:
    try:
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_LOCAL_IMAGE_BYTES:
            return None
        digest = hashlib.sha256()
        header = b""
        with path.open("rb") as handle:
            while chunk := handle.read(64 * 1024):
                if not header:
                    header = chunk[:16]
                digest.update(chunk)
        return "content:" + digest.hexdigest() if _is_image_header(header) else None
    except OSError:
        return None


def _trusted_local_path(
    path: Path,
    trusted_local_roots: tuple[Path, ...],
) -> Path | None:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    for root in trusted_local_roots:
        try:
            trusted = Path(root).expanduser().resolve(strict=False)
            if resolved == trusted or resolved.is_relative_to(trusted):
                return resolved
        except (OSError, RuntimeError, ValueError):
            continue
    return None


def _prepared(
    image: DraftImage,
    *,
    trusted_local_roots: tuple[Path, ...],
) -> PreparedImage | None:
    reference = str(image.reference or "").strip()
    if not reference or reference.startswith("base64://"):
        return None
    parsed = urlsplit(reference)
    is_windows_path = (
        len(reference) >= 3
        and reference[1] == ":"
        and reference[2] in {"/", "\\"}
    )
    if is_windows_path:
        local_path = _trusted_local_path(Path(reference), trusted_local_roots)
        fingerprint = _local_image_hash(local_path) if local_path else None
    elif parsed.scheme.casefold() in {"http", "https"} and parsed.netloc:
        fingerprint = _remote_fingerprint(reference)
    elif parsed.scheme.casefold() == "file":
        local_path = _trusted_local_path(Path(parsed.path), trusted_local_roots)
        fingerprint = _local_image_hash(local_path) if local_path else None
    elif not parsed.scheme:
        local_path = _trusted_local_path(Path(reference), trusted_local_roots)
        fingerprint = _local_image_hash(local_path) if local_path else None
    else:
        fingerprint = None
    if not fingerprint:
        return None
    return PreparedImage(
        evidence_id=image.evidence_id,
        reference=reference,
        message_evidence_id=image.message_evidence_id,
        timestamp=image.timestamp,
        fingerprint=fingerprint,
        context_weight=max(0, min(3, int(image.context_weight))),
    )


def _distributed_sample(
    images: list[PreparedImage], maximum: int
) -> list[PreparedImage]:
    limit = max(0, min(len(images), int(maximum)))
    if limit <= 0:
        return []
    if limit == 1:
        return [images[len(images) // 2]]
    if limit >= len(images):
        return images
    anchors = [
        round(position * (len(images) - 1) / (limit - 1))
        for position in range(limit)
    ]
    radius = max(0, (len(images) - 1) // max(1, 2 * (limit - 1)))
    selected: set[int] = set()
    for anchor in anchors:
        candidates = [
            index
            for index in range(max(0, anchor - radius), min(len(images), anchor + radius + 1))
            if index not in selected
        ]
        if not candidates:
            candidates = [index for index in range(len(images)) if index not in selected]
        chosen = min(
            candidates,
            key=lambda index: (-images[index].context_weight, abs(index - anchor), index),
        )
        selected.add(chosen)
    return [images[index] for index in sorted(selected)]


def prepare_images(
    images: tuple[DraftImage, ...] | list[DraftImage],
    *,
    maximum: int,
    trusted_local_roots: tuple[Path, ...] = (),
) -> ImagePreparationResult:
    """Validate references, content-dedupe local media, and sample the timeline."""

    valid: list[PreparedImage] = []
    seen: set[str] = set()
    invalid = 0
    duplicates = 0
    ordered = sorted(
        enumerate(images),
        key=lambda item: (
            item[1].timestamp is None,
            item[1].timestamp or 0,
            item[0],
        ),
    )
    for _position, image in ordered:
        prepared = _prepared(
            image,
            trusted_local_roots=trusted_local_roots,
        )
        if prepared is None:
            invalid += 1
            continue
        if prepared.fingerprint in seen:
            duplicates += 1
            continue
        seen.add(prepared.fingerprint)
        valid.append(prepared)
    return ImagePreparationResult(
        images=tuple(_distributed_sample(valid, max(1, min(20, int(maximum))))),
        invalid_count=invalid,
        duplicate_count=duplicates,
    )


def _assert_public_remote_url(reference: str) -> None:
    parsed = urlsplit(reference)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("unsupported image URL")
    if parsed.username or parsed.password:
        raise ValueError("credentials in image URL are forbidden")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("nonstandard image URL port is forbidden")
    addresses = {
        item[4][0]
        for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    }
    if not addresses:
        raise ValueError("image host did not resolve")
    for value in addresses:
        address = ipaddress.ip_address(value.split("%", 1)[0])
        if not address.is_global:
            raise ValueError("private or non-global image host is forbidden")


def _assert_public_peer(response: object) -> None:
    """Validate the connected peer after DNS resolution to block rebinding."""

    candidates = [
        getattr(getattr(response, "fp", None), "raw", None),
        getattr(
            getattr(getattr(response, "fp", None), "fp", None),
            "raw",
            None,
        ),
    ]
    peer_value = ""
    for candidate in candidates:
        sock = getattr(candidate, "_sock", None)
        if sock is None:
            continue
        try:
            peer_value = str(sock.getpeername()[0])
            break
        except (OSError, TypeError, IndexError):
            continue
    if not peer_value:
        raise ValueError("unable to validate image connection peer")
    peer = ipaddress.ip_address(peer_value.split("%", 1)[0])
    if not peer.is_global:
        raise ValueError("private or non-global image connection is forbidden")


class _PublicOnlyRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        _assert_public_remote_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _remote_content_fingerprint(
    reference: str,
    timeout_seconds: float,
    maximum_bytes: int,
) -> RemoteProbeResult | None:
    """Read a public image stream, hash it, and resize it when necessary."""

    raw_path = ""
    try:
        _assert_public_remote_url(reference)
        opener = build_opener(_PublicOnlyRedirectHandler())
        request = Request(
            reference,
            headers={
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.1",
                "User-Agent": "AstrBot-Plugin-Advisor/0.7",
            },
        )
        digest = hashlib.sha256()
        total = 0
        header = b""
        with tempfile.NamedTemporaryFile(
            prefix="astrbot-advisor-source-", suffix=".img", delete=False
        ) as raw_file:
            raw_path = raw_file.name
            with opener.open(
                request, timeout=max(1.0, min(15.0, timeout_seconds))
            ) as response:
                _assert_public_remote_url(response.geturl())
                _assert_public_peer(response)
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > maximum_bytes:
                    return None
                while chunk := response.read(
                    min(64 * 1024, maximum_bytes - total + 1)
                ):
                    total += len(chunk)
                    if total > maximum_bytes:
                        return None
                    if not header:
                        header = chunk[:16]
                    digest.update(chunk)
                    raw_file.write(chunk)
        if not total or not _is_image_header(header):
            return None
        prepared_reference, temporary_path = _resize_image_file(
            Path(raw_path), original_reference=reference
        )
        return RemoteProbeResult(
            fingerprint="content:" + digest.hexdigest(),
            byte_count=total,
            reference=prepared_reference,
            temporary_path=temporary_path,
        )
    except Exception:
        return None
    finally:
        if raw_path:
            try:
                Path(raw_path).unlink(missing_ok=True)
            except OSError:
                pass


def _resize_image_file(
    source_path: Path,
    *,
    original_reference: str,
) -> tuple[str, str]:
    """Return a bounded local copy only when pixel or file dimensions require it."""

    output_path = ""
    try:
        source_bytes = source_path.stat().st_size
        with PILImage.open(source_path) as opened:
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > MAX_DECODED_PIXELS:
                raise ValueError("image dimensions are unsafe")
            if (
                max(width, height) <= MAX_IMAGE_LONG_EDGE
                and source_bytes <= RESIZE_FILE_THRESHOLD_BYTES
            ):
                return original_reference, ""
            if opened.format == "JPEG":
                opened.draft("RGB", (MAX_IMAGE_LONG_EDGE, MAX_IMAGE_LONG_EDGE))
            opened.seek(0)
            has_alpha = opened.mode in {"RGBA", "LA"} or (
                opened.mode == "P" and "transparency" in opened.info
            )
            converted = opened.convert("RGBA" if has_alpha else "RGB")
            try:
                converted.thumbnail(
                    (MAX_IMAGE_LONG_EDGE, MAX_IMAGE_LONG_EDGE),
                    PILImage.Resampling.LANCZOS,
                )
                suffix = ".png" if has_alpha else ".jpg"
                descriptor, output_path = tempfile.mkstemp(
                    prefix="astrbot-advisor-image-", suffix=suffix
                )
                os.close(descriptor)
                if has_alpha:
                    converted.save(output_path, "PNG", optimize=True)
                else:
                    converted.save(
                        output_path,
                        "JPEG",
                        quality=85,
                        optimize=True,
                    )
            finally:
                converted.close()
        return output_path, output_path
    except Exception:
        if output_path:
            try:
                Path(output_path).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _local_path(reference: str) -> Path | None:
    parsed = urlsplit(reference)
    if parsed.scheme.casefold() == "file":
        return Path(parsed.path)
    if not parsed.scheme or (
        len(reference) >= 3
        and reference[1] == ":"
        and reference[2] in {"/", "\\"}
    ):
        return Path(reference)
    return None


RemoteProbe = Callable[
    [str, float, int], RemoteProbeResult | tuple[str, int] | None
]


def _cleanup_probe_task(task: asyncio.Task) -> None:
    """Delete a late probe result after timeout or caller cancellation."""

    try:
        result = task.result()
    except BaseException:
        return
    if isinstance(result, RemoteProbeResult) and result.temporary_path:
        try:
            Path(result.temporary_path).unlink(missing_ok=True)
        except OSError:
            pass


def _cleanup_temporary_paths(paths: list[str]) -> None:
    for value in paths:
        try:
            Path(value).unlink(missing_ok=True)
        except OSError:
            pass


async def validate_remote_images(
    result: ImagePreparationResult,
    *,
    maximum: int,
    timeout_seconds: float = 6.0,
    maximum_image_bytes: int = MAX_REMOTE_IMAGE_BYTES,
    maximum_total_bytes: int = MAX_REMOTE_TOTAL_BYTES,
    probe: RemoteProbe = _remote_content_fingerprint,
    trusted_local_roots: tuple[Path, ...] = (),
) -> ImagePreparationResult:
    """Content-dedupe images and create bounded temporary copies when needed."""

    valid: list[PreparedImage] = []
    seen: set[str] = set()
    invalid = result.invalid_count
    duplicates = result.duplicate_count
    checked = 0
    total_bytes = 0
    temporary_paths: list[str] = []
    safe_maximum = max(1, min(20, int(maximum)))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(4.0, min(20.0, timeout_seconds * 2))
    for image in result.images:
        parsed = urlsplit(image.reference)
        fingerprint = image.fingerprint
        if parsed.scheme.casefold() in {"http", "https"}:
            remaining = min(maximum_image_bytes, maximum_total_bytes - total_bytes)
            if remaining <= 0:
                invalid += 1
                continue
            remaining_time = deadline - loop.time()
            if remaining_time <= 0:
                invalid += 1
                continue
            checked += 1
            probe_task = asyncio.create_task(
                asyncio.to_thread(
                    probe,
                    image.reference,
                    min(timeout_seconds, remaining_time),
                    remaining,
                )
            )
            try:
                probed = await asyncio.wait_for(
                    asyncio.shield(probe_task),
                    timeout=max(0.1, min(timeout_seconds + 1.0, remaining_time)),
                )
            except asyncio.CancelledError:
                probe_task.add_done_callback(_cleanup_probe_task)
                _cleanup_temporary_paths(temporary_paths)
                raise
            except Exception:
                probe_task.add_done_callback(_cleanup_probe_task)
                probed = None
            if probed is None:
                invalid += 1
                continue
            if isinstance(probed, RemoteProbeResult):
                fingerprint = probed.fingerprint
                byte_count = probed.byte_count
                prepared_reference = probed.reference
                temporary_path = probed.temporary_path
            else:
                fingerprint, byte_count = probed
                prepared_reference = image.reference
                temporary_path = ""
            total_bytes += max(0, int(byte_count))
        else:
            prepared_reference = image.reference
            temporary_path = ""
            path = _local_path(image.reference)
            if path is not None:
                path = _trusted_local_path(path, trusted_local_roots)
                if path is None:
                    invalid += 1
                    continue
                try:
                    prepared_reference, temporary_path = _resize_image_file(
                        path, original_reference=image.reference
                    )
                except (OSError, ValueError):
                    invalid += 1
                    continue
        if fingerprint in seen:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)
            duplicates += 1
            continue
        seen.add(fingerprint)
        if temporary_path:
            temporary_paths.append(temporary_path)
        valid.append(
            PreparedImage(
                evidence_id=image.evidence_id,
                reference=prepared_reference,
                message_evidence_id=image.message_evidence_id,
                timestamp=image.timestamp,
                fingerprint=fingerprint,
                context_weight=image.context_weight,
            )
        )
        if len(valid) >= safe_maximum:
            break
    return ImagePreparationResult(
        images=tuple(valid),
        invalid_count=invalid,
        duplicate_count=duplicates,
        checked_remote_count=checked,
        downloaded_bytes=total_bytes,
        temporary_paths=tuple(temporary_paths),
    )


def cleanup_prepared_images(result: ImagePreparationResult | None) -> None:
    """Delete temporary resized copies. Safe to call more than once."""

    if result is None:
        return
    result.cleanup()
