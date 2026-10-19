#!/usr/bin/env python3
"""Fetch only the OXTS (GPS/IMU) folders of KITTI Raw drives.

KITTI publishes each drive as one ``<drive>_sync.zip`` of 4-21 GB (camera images, Velodyne
scans, OXTS). NavFusion needs only the OXTS text files, about 2 MB per drive. The trick: a zip
file's table of contents (the *central directory*) sits at the end of the archive and records
the byte offset of every member, and the KITTI S3 bucket honours HTTP ``Range`` requests. So
the remote zip is wrapped in a seekable file object whose reads become range requests, and the
standard ``zipfile`` module then fetches only the central directory and the ``oxts/`` members.
Across all 8 drives that is about 40 MB of transfer instead of about 91 GB.

Standard library only, so it runs before ``pip install``, both locally and in Colab.
Normally called through ``data/download_data.sh``.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
READ_BUFFER = 1 << 20  # 1 MiB: oxts members are contiguous, so few, large range reads suffice


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file. Every ``read`` is one HTTP Range request."""

    def __init__(self, url: str, retries: int = 4, timeout: float = 60.0) -> None:
        self.url, self.retries, self.timeout = url, retries, timeout
        self._pos = 0
        self.bytes_fetched = 0
        with self._open(Request(url, method="HEAD")) as resp:
            self.size = int(resp.headers["Content-Length"])

    def _open(self, req: Request):
        for attempt in range(1, self.retries + 1):
            try:
                return urlopen(req, timeout=self.timeout)
            except (URLError, TimeoutError, ConnectionError):
                if attempt == self.retries:
                    raise
                time.sleep(2**attempt)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = base + offset
        return self._pos

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self._pos)
        if n <= 0:
            return 0
        req = Request(self.url, headers={"Range": f"bytes={self._pos}-{self._pos + n - 1}"})
        with self._open(req) as resp:
            # A 200 here would mean the server ignored Range and is streaming the whole
            # multi-GB zip. Refuse rather than silently fall back to the full download.
            if resp.status != 206:
                raise RuntimeError(
                    f"{self.url}: server ignored the Range header (HTTP {resp.status})"
                )
            data = resp.read()
        if len(data) != n:
            raise OSError(f"{self.url}: short range read ({len(data)} of {n} bytes)")
        buf[:n] = data
        self._pos += n
        self.bytes_fetched += n
        return n


def fetch_drive_oxts(drive: str, out_root: Path) -> None:
    """Extract ``<date>/<drive>_sync/oxts/**`` from the remote sync zip into ``out_root``."""
    date = drive[:10]
    prefix = f"{date}/{drive}_sync/oxts/"
    dest = out_root / date / f"{drive}_sync" / "oxts"
    if dest.is_dir():
        print(f"  {drive}: already present, skipping")
        return

    remote = HttpRangeFile(f"{BASE_URL}/{drive}/{drive}_sync.zip")
    # Extract into a sibling temp dir and rename at the end, so an interrupted run never
    # leaves a half-filled oxts/ that a later run would mistake for a complete drive.
    tmp = dest.with_name("oxts.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.monotonic()
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=READ_BUFFER)) as zf:
        members = [m for m in zf.infolist() if m.filename.startswith(prefix) and not m.is_dir()]
        if not members:
            raise RuntimeError(f"{drive}: no '{prefix}' entries in the remote zip")
        for member in members:
            rel = PurePosixPath(member.filename[len(prefix) :])
            if ".." in rel.parts or rel.is_absolute():
                raise RuntimeError(f"{drive}: unsafe path in zip: {member.filename}")
            target = tmp.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    tmp.rename(dest)
    print(
        f"  {drive}: {len(members)} files, fetched {remote.bytes_fetched / 1e6:.1f} MB "
        f"of a {remote.size / 1e9:.1f} GB zip in {time.monotonic() - t0:.0f} s"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("drives", nargs="+", help="drive names, e.g. 2011_09_30_drive_0018")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/kitti_raw"),
        help="output root (default: data/kitti_raw)",
    )
    args = parser.parse_args(argv)

    for drive in args.drives:
        fetch_drive_oxts(drive, args.out)
    print(f"Done. KITTI OXTS data is in {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
