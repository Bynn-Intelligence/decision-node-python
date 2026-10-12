"""Check the built package before it can be published.

Run from the repository root after `uv build`, with the version the files must carry in
EXPECTED_VERSION (the tag without its v on a tag, else the CHANGELOG's top entry). It
lists both files, then fails on: a name, version, license, readme or repository URL that
is not what is published; a file outside the allowlist; any file that holds an internal
reference or a word that must not ship; and a README or description that names a booster
production does not serve yet.
"""

from __future__ import annotations

import email.parser
import hashlib
import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

NAME = "bynn-decisionnode"
DIST = NAME.replace("-", "_")
FORBIDDEN = re.compile(
    r"docs/[0-9]{2}|backend/[0-9]{2}|products/|/Users/|co-authored-by|generated with",
    re.IGNORECASE,
)
# Words that must not ship, matched case-insensitively, kept as SHA-256 digests so this
# repository does not spell them: four found anywhere in a word, one only as a whole word.
INSIDE_WORDS = {
    "9e9b2efac0b62e4cf6f08cc6b7095e6ad7e18c447458c5849accb698f7149ef3",
    "934a11e602682e862b4b4d1f881ae8da1e097e3f6964bd780e5bb1fb798a6fdd",
    "b7933ed6180d9e2895a9d2df885e2a8b92e998a3c16297581f5b8698daacf492",
    "31d1b3d540ecf85228db7797efbe7d198ec5406abd3f6e64f27f684e9a312d29",
}
INSIDE_LENGTHS = (6, 7)
WHOLE_WORDS = {"2e0b8d61fa2a6959d254b6ff5d0fb512249329097336a35568089933b49abdde"}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


UNSERVED = re.compile(r"face_detection|wanted_person")
PACKAGE_FILE = re.compile(r"[A-Za-z0-9_]+\.py|py\.typed")


def fail(errors: List[str]) -> None:
    for e in errors:
        print("::error::%s" % e)
    sys.exit(1)


def metadata(text: str) -> Tuple[Dict[str, List[str]], str]:
    msg = email.parser.Parser().parsestr(text)
    fields: Dict[str, List[str]] = {}
    for key, value in msg.items():
        fields.setdefault(key.lower(), []).append(value)
    return fields, msg.get_payload() or ""


def check_metadata(where: str, text: str, version: str, errors: List[str]) -> str:
    fields, body = metadata(text)

    def one(key: str) -> str:
        return (fields.get(key) or [""])[0]

    if one("name") != NAME:
        errors.append("%s: name %r, expected %r" % (where, one("name"), NAME))
    if one("version") != version:
        errors.append("%s: version %r, expected %r" % (where, one("version"), version))
    classifiers = fields.get("classifier", [])
    if not (one("license") or one("license-expression") or any("License ::" in c for c in classifiers)):
        errors.append("%s: no license" % where)
    if one("description-content-type") != "text/markdown" or not body.strip():
        errors.append("%s: no README as the long description" % where)
    urls = fields.get("project-url", [])
    if not any(u.lower().startswith("repository,") and "github.com/Bynn-Intelligence/" in u for u in urls):
        errors.append("%s: no repository URL (Project-URL: Repository, ...)" % where)
    if UNSERVED.search(body) or UNSERVED.search(one("summary")):
        errors.append("%s: the README or the summary names face_detection or wanted_person" % where)
    return body


def scan(where: str, name: str, data: bytes, errors: List[str]) -> None:
    text = data.decode("utf-8", errors="replace")

    def found(start: int, what: str) -> None:
        line = text.count("\n", 0, start) + 1
        errors.append("%s: %s line %d contains %s" % (where, name, line, what))

    for m in FORBIDDEN.finditer(text):
        found(m.start(), repr(m.group(0)))
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


def check_wheel(path: Path, version: str, errors: List[str]) -> None:
    info = "%s-%s.dist-info/" % (DIST, version)
    allowed = {info + n for n in ("METADATA", "WHEEL", "RECORD", "licenses/LICENSE")}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        print("%s:" % path.name)
        for n in names:
            print("  %s" % n)
            ok = n in allowed or (n.startswith("decisionnode/") and PACKAGE_FILE.fullmatch(n[len("decisionnode/") :]))
            if not ok:
                errors.append("%s: %s is not in the allowlist" % (path.name, n))
            scan(path.name, n, z.read(n), errors)
        if info + "METADATA" not in names:
            errors.append("%s: no METADATA" % path.name)
        else:
            check_metadata(path.name, z.read(info + "METADATA").decode("utf-8"), version, errors)


def check_sdist(path: Path, version: str, errors: List[str]) -> None:
    root = "%s-%s/" % (DIST, version)
    top = {root + n for n in ("pyproject.toml", "README.md", "LICENSE", "CHANGELOG.md", "PKG-INFO", ".gitignore")}
    with tarfile.open(path, "r:gz") as t:
        members = [m for m in t.getmembers() if m.isfile()]
        print("%s:" % path.name)
        files = {}
        for m in members:
            print("  %s" % m.name)
            rest = m.name[len(root) :] if m.name.startswith(root) else None
            ok = m.name in top or (
                rest is not None
                and rest.startswith("src/decisionnode/")
                and PACKAGE_FILE.fullmatch(rest[len("src/decisionnode/") :])
            )
            if not ok:
                errors.append("%s: %s is not in the allowlist" % (path.name, m.name))
            f = t.extractfile(m)
            data = f.read() if f else b""
            files[m.name] = data
            scan(path.name, m.name, data, errors)
        if root + "PKG-INFO" not in files:
            errors.append("%s: no PKG-INFO" % path.name)
        else:
            check_metadata(path.name, files[root + "PKG-INFO"].decode("utf-8"), version, errors)
        readme = files.get(root + "README.md", b"").decode("utf-8", errors="replace")
        if not readme.strip():
            errors.append("%s: no README.md" % path.name)
        elif UNSERVED.search(readme):
            errors.append("%s: README.md names face_detection or wanted_person" % path.name)


def main() -> None:
    version = os.environ.get("EXPECTED_VERSION", "")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+([.-]?[0-9A-Za-z.]+)?", version):
        fail(["EXPECTED_VERSION %r is not a version" % version])
    dist = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    # uv build writes a .gitignore into dist/; it is never uploaded.
    built_files = {p.name for p in wheels + sdists} | {".gitignore"}
    others = sorted(p.name for p in dist.iterdir() if p.is_file() and p.name not in built_files)
    errors: List[str] = []
    if len(wheels) != 1 or len(sdists) != 1:
        errors.append("expected one wheel and one sdist in %s, found %d and %d" % (dist, len(wheels), len(sdists)))
    if others:
        errors.append("unexpected files in %s: %s" % (dist, ", ".join(others)))
    # The files' own version decides the paths inside them; it must be the expected one.
    built = {re.sub(r"^%s-([^-]+)(-.*\.whl|\.tar\.gz)$" % DIST, r"\1", p.name) for p in wheels + sdists}
    if built != {version}:
        errors.append("the built files are version %s, expected %s" % (", ".join(sorted(built)), version))
    if len(built) == 1:
        (inside,) = built
        for w in wheels:
            check_wheel(w, inside, errors)
        for s in sdists:
            check_sdist(s, inside, errors)
    if errors:
        fail(errors)
    print("ok: %s %s, %d file(s) checked" % (NAME, version, len(wheels) + len(sdists)))


if __name__ == "__main__":
    main()
