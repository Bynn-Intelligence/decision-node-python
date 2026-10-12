"""Check the files a release is about to publish.

Usage: python3 check_names.py <file or directory>...

Reads every file given, every file under a directory, and every member of an archive
(.whl, .zip, .tar, .tar.gz, .tgz, .gem, nested archives and .gz members included), and
fails when one holds an internal reference or a word that must not ship, or when a
README or the package's description names a booster production does not serve yet. On
a hit it prints the file and line, never the word.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator, List, Tuple

FORBIDDEN = re.compile(
    r"docs/[0-9]{2}|backend/[0-9]{2}|products/|/Users/|co-authored-by|generated with",
    re.IGNORECASE,
)
# Words that must not ship, matched case-insensitively and kept as SHA-256 digests so
# this repository does not spell them: four found anywhere in a word, one only as a
# whole word.
INSIDE_WORDS = {
    "9e9b2efac0b62e4cf6f08cc6b7095e6ad7e18c447458c5849accb698f7149ef3",
    "934a11e602682e862b4b4d1f881ae8da1e097e3f6964bd780e5bb1fb798a6fdd",
    "b7933ed6180d9e2895a9d2df885e2a8b92e998a3c16297581f5b8698daacf492",
    "31d1b3d540ecf85228db7797efbe7d198ec5406abd3f6e64f27f684e9a312d29",
}
INSIDE_LENGTHS = (6, 7)
WHOLE_WORDS = {"2e0b8d61fa2a6959d254b6ff5d0fb512249329097336a35568089933b49abdde"}
ARCHIVE = re.compile(r"\.(whl|zip|tar|tar\.gz|tgz|gem)$")
# Boosters a README may name only once production serves them, and the files that carry
# a README or the package's description: a README itself, a wheel's METADATA or an sdist's
# PKG-INFO, an npm package.json, a gem's metadata.
UNSERVED = re.compile(r"face_detection|wanted_person|face_recognition")
DESCRIBES = re.compile(r"(^|/)(README[^/]*|PKG-INFO|METADATA|package\.json|metadata)$", re.IGNORECASE)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def members(name: str, data: bytes) -> Iterator[Tuple[str, bytes]]:
    """The files inside an archive, nested archives opened in turn; a plain file as is."""
    if name.endswith((".whl", ".zip")):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                if not info.is_dir():
                    yield from members("%s!%s" % (name, info.filename), z.read(info))
    elif ARCHIVE.search(name):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
            for m in t.getmembers():
                if m.isfile():
                    f = t.extractfile(m)
                    if f is not None:
                        yield from members("%s!%s" % (name, m.name), f.read())
    elif name.endswith(".gz"):
        yield from members(name[: -len(".gz")], gzip.decompress(data))
    else:
        yield name, data


def scan(name: str, data: bytes) -> List[str]:
    text = data.decode("utf-8", errors="replace")
    errors: List[str] = []

    def found(start: int, what: str) -> None:
        errors.append("%s line %d contains %s" % (name, text.count("\n", 0, start) + 1, what))

    for m in FORBIDDEN.finditer(text):
        found(m.start(), repr(m.group(0)))
    if DESCRIBES.search(name.split("!")[-1]):
        for m in UNSERVED.finditer(text):
            found(m.start(), "%r, a booster production does not serve yet" % m.group(0))
    lower = text.lower()
    for m in re.finditer(r"[a-z]{%d,}" % min(INSIDE_LENGTHS), lower):
        run = m.group(0)
        for size in INSIDE_LENGTHS:
            for i in range(len(run) - size + 1):
                if digest(run[i : i + size]) in INSIDE_WORDS:
                    found(m.start() + i, "a word that must not ship")
    for m in re.finditer(r"\w+", lower):
        if digest(m.group(0)) in WHOLE_WORDS:
            found(m.start(), "a word that must not ship")
    return errors


def main(argv: List[str]) -> int:
    if not argv:
        print("usage: check_names.py <file or directory>...", file=sys.stderr)
        return 2
    paths: List[Path] = []
    for arg in argv:
        p = Path(arg)
        if not p.exists():
            print("::error::%s does not exist" % arg)
            return 1
        paths.extend(sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p])
    errors: List[str] = []
    scanned = 0
    for p in paths:
        for name, data in members(str(p), p.read_bytes()):
            scanned += 1
            errors.extend(scan(name, data))
    for e in errors:
        print("::error::%s" % e)
    print("%d files scanned, %d findings" % (scanned, len(errors)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
